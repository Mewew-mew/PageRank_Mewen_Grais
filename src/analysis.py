"""Analyse des résultats PR / PPR, diagnostic de seuil et calcul du ratio PPR / PR.

Ce module produit :
1. Le Top K (défaut 20) du PageRank classique (PR) enrichi des titres Wikipedia.
2. Le Top K (défaut 20) du PageRank personnalisé depuis Nantes (PPR).
3. Un diagnostic statistique complet de la distribution du PPR (quantiles + grille
   de seuils candidats : 1/N, 5/N, 10/N, 1e-5, 1e-4) pour choisir et justifier
   rigoureusement le seuil minimal avant de calculer `PPR(p) / PR(p)`.
4. Le Top K (défaut 20) du rapport `PPR(p) / PR(p)` filtré sur `PPR(p) >= seuil`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import COL_PAGE_ID, COL_PAGE_TITLE, COL_RANK, DEFAULT_TOP_K


def compute_top_k(
    ranks_df: DataFrame,
    titles_df: DataFrame,
    k: int = DEFAULT_TOP_K,
    score_col: str = "score",
) -> pd.DataFrame:
    """Retourne le Top K des pages triées par score décroissant avec leurs titres lisibles."""
    top_df = (
        ranks_df.orderBy(F.col(COL_RANK).desc(), F.col(COL_PAGE_ID).asc())
        .limit(k)
        .join(titles_df, on=COL_PAGE_ID, how="left")
        .select(
            F.col(COL_PAGE_ID),
            F.coalesce(F.col(COL_PAGE_TITLE), F.lit("<unknown>")).alias(COL_PAGE_TITLE),
            F.col(COL_RANK).alias(score_col),
        )
        .orderBy(F.col(score_col).desc(), F.col(COL_PAGE_ID).asc())
    )

    rows = [row.asDict() for row in top_df.collect()]
    pdf = pd.DataFrame(rows)
    if not pdf.empty:
        pdf.insert(0, "rank_pos", range(1, len(pdf) + 1))
    return pdf


def analyze_ppr_distribution(
    pr_df: DataFrame,
    ppr_df: DataFrame,
    num_nodes: int,
    extra_thresholds: Optional[list[float]] = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Produit les quantiles du PPR et une table de sensibilité multi-seuils.

    Pourquoi un seuil est indispensable pour `PPR(p) / PR(p)` ?
    Sans seuil, des pages quasi-orphelines ou très périphériques possédant un `PR(p)`
    proche du plancher `(1 - d) / N` mais un faible lien indirect depuis Nantes peuvent
    afficher un ratio artificiellement énorme tout en ayant une probabilité de visite
    (`PPR`) négligeable.

    Returns:
        Tuple `(quantiles_dict, thresholds_table_pdf)` :
        - `quantiles_dict` : quantiles (p50, p75, p90, p95, p99, p999, max) parmi les pages `ppr > 0`.
        - `thresholds_table_pdf` : DataFrame Pandas comparant chaque seuil candidat
          (`1/N`, `5/N`, `10/N`, `1e-5`, `1e-4`, etc.) en nombre de pages retenues,
          part du graphe, masse PPR couverte et ratio max.
    """
    joined = (
        pr_df.select(F.col(COL_PAGE_ID), F.col(COL_RANK).alias("pr"))
        .join(
            ppr_df.select(F.col(COL_PAGE_ID), F.col(COL_RANK).alias("ppr")),
            on=COL_PAGE_ID,
            how="inner",
        )
        .withColumn(
            "ratio",
            F.when(F.col("pr") > F.lit(0.0), F.col("ppr") / F.col("pr")).otherwise(F.lit(0.0)),
        )
    )

    # 1. Quantiles du PPR parmi les pages effectivement atteintes (ppr > 0)
    probs = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]
    positive_ppr = joined.filter(F.col("ppr") > F.lit(0.0))
    q_values = positive_ppr.approxQuantile("ppr", probs, 1e-4)
    labels = ["p50", "p75", "p90", "p95", "p99", "p999", "max"]
    quantiles_dict = {
        label: float(val) for label, val in zip(labels, q_values)
    } if q_values else {}

    # 2. Grille de seuils candidats : multiples de 1/N (seuil uniforme) + seuils absolus
    uniform = 1.0 / float(num_nodes)
    candidates: list[tuple[str, float]] = [
        ("1/N (uniforme)", uniform),
        ("5/N", 5.0 * uniform),
        ("10/N", 10.0 * uniform),
        ("1e-5", 1e-5),
        ("1e-4", 1e-4),
    ]
    if extra_thresholds:
        for val in extra_thresholds:
            candidates.append((f"custom ({val:.2e})", float(val)))

    # Calcul des statistiques en UNE seule agrégation multi-colonnes sur `joined`
    agg_exprs = []
    for idx, (_, thr) in enumerate(candidates):
        cond = F.col("ppr") >= F.lit(thr)
        agg_exprs.extend(
            [
                F.sum(F.when(cond, F.lit(1)).otherwise(F.lit(0))).alias(f"cnt_{idx}"),
                F.sum(F.when(cond, F.col("ppr")).otherwise(F.lit(0.0))).alias(f"mass_{idx}"),
                F.max(F.when(cond, F.col("ratio"))).alias(f"max_ratio_{idx}"),
            ]
        )

    summary_row = joined.agg(*agg_exprs).first()

    records = []
    for idx, (label, thr) in enumerate(candidates):
        pages_kept = int(summary_row[f"cnt_{idx}"] or 0)
        ppr_mass = float(summary_row[f"mass_{idx}"] or 0.0)
        max_ratio = summary_row[f"max_ratio_{idx}"]
        records.append(
            {
                "threshold_label": label,
                "ppr_threshold": thr,
                "pages_kept": pages_kept,
                "pct_of_graph": (100.0 * pages_kept / float(num_nodes)) if num_nodes > 0 else 0.0,
                "ppr_mass_covered": ppr_mass,
                "max_ratio_ppr_pr": float(max_ratio) if max_ratio is not None else 0.0,
            }
        )

    return quantiles_dict, pd.DataFrame(records)


def compute_top_ratio_ppr_pr(
    pr_df: DataFrame,
    ppr_df: DataFrame,
    titles_df: DataFrame,
    ppr_threshold: float,
    k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Calcule le Top K du ratio `PPR(p) / PR(p)` en écartant les pages où `PPR(p) < ppr_threshold`."""
    ratio_df = (
        pr_df.select(F.col(COL_PAGE_ID), F.col(COL_RANK).alias("pr"))
        .join(
            ppr_df.select(F.col(COL_PAGE_ID), F.col(COL_RANK).alias("ppr")),
            on=COL_PAGE_ID,
            how="inner",
        )
        .filter((F.col("ppr") >= F.lit(float(ppr_threshold))) & (F.col("pr") > F.lit(0.0)))
        .withColumn("ratio_ppr_pr", F.col("ppr") / F.col("pr"))
        .orderBy(F.col("ratio_ppr_pr").desc(), F.col("ppr").desc())
        .limit(k)
        .join(titles_df, on=COL_PAGE_ID, how="left")
        .select(
            F.col(COL_PAGE_ID),
            F.coalesce(F.col(COL_PAGE_TITLE), F.lit("<unknown>")).alias(COL_PAGE_TITLE),
            F.col("ppr"),
            F.col("pr"),
            F.col("ratio_ppr_pr"),
        )
        .orderBy(F.col("ratio_ppr_pr").desc(), F.col("ppr").desc())
    )

    rows = [row.asDict() for row in ratio_df.collect()]
    pdf = pd.DataFrame(rows)
    if not pdf.empty:
        pdf.insert(0, "rank_pos", range(1, len(pdf) + 1))
        pdf["ppr_threshold_used"] = float(ppr_threshold)
    return pdf


def save_results_csv(pdf: pd.DataFrame, output_path: str | Path) -> Path:
    """Sauvegarde une petite table d'analyse (Top 20, diagnostic seuils) en CSV UTF-8 propre."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(path, index=False, encoding="utf-8")
    return path
