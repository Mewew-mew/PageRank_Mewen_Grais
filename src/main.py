"""Point d'entrée CLI orchestrant préparation, échantillonnage, PageRank/PPR et chronométrage.

Exemples d'utilisation :
  # 1. Conversion .csv.gz -> Parquet partitionné (1 seule lecture mono-cœur)
  python -m src.main prepare --raw-input gs://BUCKET/raw/enwiki.wikilink_graph.2018-03-01.csv.gz \
      --prepared-dir gs://BUCKET/prepared/2018_full --num-partitions 96

  # 2. Création du sous-graphe induit 10 % centré sur Nantes (67453)
  python -m src.main sample --prepared-dir gs://BUCKET/prepared/2018_full \
      --sample-dir gs://BUCKET/prepared/2018_sample10 --num-partitions 96

  # 3. Calcul PR + PPR + analyse + benchmarks + capture du plan explain()
  python -m src.main run --prepared-dir gs://BUCKET/prepared/2018_full \
      --results-dir gs://BUCKET/results/2018_full_6nodes --cluster-nodes 6 \
      --num-partitions 96 --dataset-label full_2018
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.analysis import (
    analyze_ppr_distribution,
    compute_top_k,
    compute_top_ratio_ppr_pr,
    save_results_csv,
)
from src.config import (
    DAMPING_FACTOR,
    DEFAULT_NUM_PARTITIONS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_FRACTION,
    DEFAULT_TOP_K,
    NANTES_PAGE_ID,
    NUM_ITERATIONS,
    SparkConfig,
    build_spark_session,
)
from src.data_prep import (
    create_induced_sample_parquet,
    load_prepared_graph,
    prepare_parquet_from_raw,
)
from src.pagerank import prepare_graph_tables, run_pagerank


def _is_gcs_uri(path: str) -> bool:
    return path.startswith("gs://")


def _sync_local_dir_to_gcs(local_dir: Path, gcs_dir: str) -> None:
    """Copie tous les fichiers de résultats générés localement vers un répertoire GCS."""
    target = gcs_dir.rstrip("/")
    for file_path in sorted(local_dir.iterdir()):
        if file_path.is_file():
            dest = f"{target}/{file_path.name}"
            cmd = ["gcloud", "storage", "cp", str(file_path), dest]
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                subprocess.run(["gsutil", "cp", str(file_path), dest], check=True)


def _append_benchmark_row(csv_path: Path, row: dict[str, object]) -> None:
    """Ajoute une ligne de benchmark au fichier `benchmarks.csv` (création si absent)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        new_df.to_csv(csv_path, index=False, encoding="utf-8")


def _add_common_spark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Master Spark (ex. 'local[2]' en local ; laisser vide sur Dataproc/YARN).",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=DEFAULT_NUM_PARTITIONS,
        help=f"Nombre fixe K de partitions pour le co-partitionnement (défaut: {DEFAULT_NUM_PARTITIONS}).",
    )
    parser.add_argument(
        "--enable-aqe",
        action="store_true",
        help="Active l'AQE (déconseillé : uniquement pour contre-expérience).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Répertoire HDFS/GCS requis uniquement si --checkpoint-mode=reliable.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrateur PySpark DataFrame pour Wikipedia PageRank & PPR (Nantes)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Sous-commande 1 : prepare (.csv.gz -> Parquet) ---
    p_prep = subparsers.add_parser(
        "prepare",
        help="Convertit le .csv.gz brut (mono-cœur) en Parquet découpable et partitionné.",
    )
    p_prep.add_argument("--raw-input", required=True, help="Chemin local ou gs:// du .csv.gz brut.")
    p_prep.add_argument("--prepared-dir", required=True, help="Dossier de sortie Parquet.")
    _add_common_spark_args(p_prep)

    # --- Sous-commande 2 : sample (sous-graphe induit 10 %) ---
    p_sample = subparsers.add_parser(
        "sample",
        help="Construit le sous-graphe induit 10 %% centré sur Nantes (67453) + voisinage 1-saut.",
    )
    p_sample.add_argument("--prepared-dir", required=True, help="Dossier Parquet complet en entrée.")
    p_sample.add_argument("--sample-dir", required=True, help="Dossier Parquet de sortie pour le sample.")
    p_sample.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help=f"Fraction de sommets échantillonnés (défaut: {DEFAULT_SAMPLE_FRACTION}).",
    )
    p_sample.add_argument(
        "--anchor-page-id",
        type=int,
        default=NANTES_PAGE_ID,
        help=f"Identifiant de la page ancre (défaut Nantes: {NANTES_PAGE_ID}).",
    )
    p_sample.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Graine aléatoire fixe (défaut: {DEFAULT_RANDOM_SEED}).",
    )
    _add_common_spark_args(p_sample)

    # --- Sous-commande 3 : run (PR + PPR + analyse + benchmarks) ---
    p_run = subparsers.add_parser(
        "run",
        help="Exécute PR classique + PPR Nantes sur un dataset Parquet préparé et exporte les résultats.",
    )
    p_run.add_argument("--prepared-dir", required=True, help="Dossier Parquet (full ou sample10).")
    p_run.add_argument(
        "--results-dir",
        default="results",
        help="Dossier local ou gs:// pour sauvegarder les CSV, explain et benchmarks.",
    )
    p_run.add_argument(
        "--dataset-label",
        default="full_2018",
        help="Étiquette du dataset (ex. 'sample10_2018', 'full_2018', 'full_2008').",
    )
    p_run.add_argument(
        "--cluster-nodes",
        type=int,
        default=1,
        help="Nombre de workers du cluster (2, 4 ou 6 sur Dataproc ; 1 en local).",
    )
    p_run.add_argument(
        "--source-page-id",
        type=int,
        default=NANTES_PAGE_ID,
        help=f"Identifiant de la page source PPR (défaut Nantes: {NANTES_PAGE_ID}).",
    )
    p_run.add_argument(
        "--damping",
        type=float,
        default=DAMPING_FACTOR,
        help=f"Facteur d'amortissement d (défaut: {DAMPING_FACTOR}).",
    )
    p_run.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=f"Nombre exact d'itérations (défaut: {NUM_ITERATIONS}).",
    )
    p_run.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Nombre de pages dans les Top-K (défaut: {DEFAULT_TOP_K}).",
    )
    p_run.add_argument(
        "--ppr-threshold",
        type=float,
        default=None,
        help="Seuil minimal PPR pour le ratio PPR/PR (défaut: 10/N, justifié par le diagnostic).",
    )
    p_run.add_argument(
        "--checkpoint-mode",
        choices=["local", "reliable"],
        default="local",
        help="Méthode de coupure de lignée : 'local' (localCheckpoint) ou 'reliable' (checkpoint HDFS/GCS).",
    )
    p_run.add_argument(
        "--skip-exchange-guard",
        action="store_true",
        help="Désactive l'arrêt en cas d'Exchange != 1 (utile seulement avec --enable-aqe).",
    )
    _add_common_spark_args(p_run)

    return parser


def _run_prepare(args: argparse.Namespace) -> None:
    spark = build_spark_session(
        SparkConfig(
            app_name="Wikipedia-PageRank-Prepare",
            num_partitions=args.num_partitions,
            master=args.master,
            disable_aqe=not args.enable_aqe,
            checkpoint_dir=args.checkpoint_dir,
        )
    )
    try:
        metrics = prepare_parquet_from_raw(
            spark=spark,
            raw_tsv_path=args.raw_input,
            output_dir=args.prepared_dir,
            num_partitions=args.num_partitions,
        )
        print(
            f"[PREPARE] Terminé en {metrics.raw_read_and_write_seconds:.2f}s | "
            f"Arêtes propres: {metrics.num_edges:,} | Titres: {metrics.num_titles:,}\n"
            f"  -> {metrics.edges_path}\n"
            f"  -> {metrics.titles_path}"
        )
    finally:
        spark.stop()


def _run_sample(args: argparse.Namespace) -> None:
    spark = build_spark_session(
        SparkConfig(
            app_name="Wikipedia-PageRank-Sample",
            num_partitions=args.num_partitions,
            master=args.master,
            disable_aqe=not args.enable_aqe,
            checkpoint_dir=args.checkpoint_dir,
        )
    )
    try:
        metrics = create_induced_sample_parquet(
            spark=spark,
            prepared_dir=args.prepared_dir,
            sample_output_dir=args.sample_dir,
            num_partitions=args.num_partitions,
            sample_fraction=args.sample_fraction,
            anchor_page_id=args.anchor_page_id,
            seed=args.seed,
        )
        print(
            f"[SAMPLE] Sous-graphe induit ({args.sample_fraction:.0%} + ancre {args.anchor_page_id}) "
            f"créé en {metrics.raw_read_and_write_seconds:.2f}s | "
            f"Arêtes: {metrics.num_edges:,} | Titres: {metrics.num_titles:,}\n"
            f"  -> {metrics.edges_path}\n"
            f"  -> {metrics.titles_path}"
        )
    finally:
        spark.stop()


def _run_experiment(args: argparse.Namespace) -> None:
    if args.checkpoint_mode == "reliable" and not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir est obligatoire avec --checkpoint-mode=reliable.")

    spark = build_spark_session(
        SparkConfig(
            app_name=f"Wikipedia-PageRank-{args.dataset_label}-{args.cluster_nodes}w",
            num_partitions=args.num_partitions,
            master=args.master,
            disable_aqe=not args.enable_aqe,
            disable_auto_broadcast=True,
            checkpoint_dir=args.checkpoint_dir,
        )
    )

    try:
        # 1. Phase de préparation en mémoire (lecture Parquet + degrés + co-partitionnement K)
        t_prep0 = time.perf_counter()
        edges_df, titles_df = load_prepared_graph(spark, args.prepared_dir)
        links_df, base_nodes_df, num_nodes, num_edges, num_dangling = prepare_graph_tables(
            edges_df=edges_df,
            num_partitions=args.num_partitions,
        )
        prep_graph_seconds = time.perf_counter() - t_prep0
        print(
            f"[GRAPH] Chargé & co-partitionné (K={args.num_partitions}) en {prep_graph_seconds:.2f}s | "
            f"N={num_nodes:,} sommets | M={num_edges:,} arêtes | Dangling={num_dangling:,}"
        )

        # 2. PageRank classique (PR)
        pr_res = run_pagerank(
            links_df=links_df,
            base_nodes_df=base_nodes_df,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_dangling=num_dangling,
            source_page_id=None,
            damping_factor=args.damping,
            num_iterations=args.iterations,
            checkpoint_mode=args.checkpoint_mode,
            enforce_single_exchange=not args.skip_exchange_guard,
        )
        pr_total_s = sum(pr_res.iteration_times)
        pr_avg_s = statistics.mean(pr_res.iteration_times)
        print(
            f"[PR]  {args.iterations} itérations en {pr_total_s:.2f}s "
            f"(moy={pr_avg_s:.2f}s/iter, min={min(pr_res.iteration_times):.2f}s, "
            f"max={max(pr_res.iteration_times):.2f}s) | Exchanges={pr_res.exchange_count} | "
            f"sum(r_20)={pr_res.rank_sums[-1]:.8f}"
        )

        # 3. PageRank personnalisé (PPR depuis Nantes)
        ppr_res = run_pagerank(
            links_df=links_df,
            base_nodes_df=base_nodes_df,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_dangling=num_dangling,
            source_page_id=args.source_page_id,
            damping_factor=args.damping,
            num_iterations=args.iterations,
            checkpoint_mode=args.checkpoint_mode,
            enforce_single_exchange=not args.skip_exchange_guard,
        )
        ppr_total_s = sum(ppr_res.iteration_times)
        ppr_avg_s = statistics.mean(ppr_res.iteration_times)
        print(
            f"[PPR] {args.iterations} itérations en {ppr_total_s:.2f}s "
            f"(moy={ppr_avg_s:.2f}s/iter, min={min(ppr_res.iteration_times):.2f}s, "
            f"max={max(ppr_res.iteration_times):.2f}s) | Exchanges={ppr_res.exchange_count} | "
            f"sum(r_20)={ppr_res.rank_sums[-1]:.8f}"
        )

        # 4. Analyses : Top 20 PR, Top 20 PPR, diagnostic seuils, Top 20 PPR/PR
        ppr_threshold: float = (
            float(args.ppr_threshold)
            if args.ppr_threshold is not None
            else (10.0 / float(num_nodes))
        )
        extra_thr = [ppr_threshold] if args.ppr_threshold is not None else None

        top_pr_pdf = compute_top_k(pr_res.ranks, titles_df, k=args.top_k, score_col="pr")
        top_ppr_pdf = compute_top_k(ppr_res.ranks, titles_df, k=args.top_k, score_col="ppr")
        quantiles, thresholds_pdf = analyze_ppr_distribution(
            pr_res.ranks, ppr_res.ranks, num_nodes=num_nodes, extra_thresholds=extra_thr
        )
        top_ratio_pdf = compute_top_ratio_ppr_pr(
            pr_res.ranks,
            ppr_res.ranks,
            titles_df,
            ppr_threshold=ppr_threshold,
            k=args.top_k,
        )

        print(f"[ANALYSIS] Quantiles PPR (ppr > 0) : {quantiles}")
        print(f"[ANALYSIS] Seuil PPR utilisé pour PPR/PR : {ppr_threshold:.3e}")

        # 5. Sauvegarde des artefacts (localement ou synchronisés vers gs://)
        use_gcs = _is_gcs_uri(args.results_dir)
        temp_ctx: Optional[tempfile.TemporaryDirectory[str]] = (
            tempfile.TemporaryDirectory(prefix="pagerank_results_") if use_gcs else None
        )
        out_dir = Path(temp_ctx.name) if temp_ctx is not None else Path(args.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            (out_dir / "explain_iteration.txt").write_text(pr_res.explain_plan, encoding="utf-8")
            save_results_csv(top_pr_pdf, out_dir / "top20_pr.csv")
            save_results_csv(top_ppr_pdf, out_dir / "top20_ppr_nantes.csv")
            save_results_csv(thresholds_pdf, out_dir / "ppr_threshold_diagnostics.csv")
            save_results_csv(top_ratio_pdf, out_dir / "top20_ratio_ppr_pr.csv")

            benchmark_row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset_label": args.dataset_label,
                "cluster_nodes": args.cluster_nodes,
                "num_partitions": args.num_partitions,
                "checkpoint_mode": args.checkpoint_mode,
                "aqe_enabled": bool(args.enable_aqe),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_dangling": num_dangling,
                "prep_graph_s": round(prep_graph_seconds, 3),
                "pr_total_s": round(pr_total_s, 3),
                "pr_avg_iter_s": round(pr_avg_s, 3),
                "pr_min_iter_s": round(min(pr_res.iteration_times), 3),
                "pr_max_iter_s": round(max(pr_res.iteration_times), 3),
                "ppr_total_s": round(ppr_total_s, 3),
                "ppr_avg_iter_s": round(ppr_avg_s, 3),
                "ppr_min_iter_s": round(min(ppr_res.iteration_times), 3),
                "ppr_max_iter_s": round(max(ppr_res.iteration_times), 3),
                "exchange_count_pr": pr_res.exchange_count,
                "exchange_count_ppr": ppr_res.exchange_count,
                "ppr_threshold_used": ppr_threshold,
            }
            _append_benchmark_row(out_dir / "benchmarks.csv", benchmark_row)

            if use_gcs:
                _sync_local_dir_to_gcs(out_dir, args.results_dir)
                print(f"[OUTPUT] Tous les résultats ont été synchronisés vers {args.results_dir}")
            else:
                print(f"[OUTPUT] Tous les résultats ont été sauvegardés dans {out_dir.resolve()}")
        finally:
            if temp_ctx is not None:
                temp_ctx.cleanup()

        links_df.unpersist()
        base_nodes_df.unpersist()
    finally:
        spark.stop()


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        _run_prepare(args)
    elif args.command == "sample":
        _run_sample(args)
    elif args.command == "run":
        _run_experiment(args)
    else:
        parser.error(f"Commande inconnue : {args.command}")


if __name__ == "__main__":
    main()
