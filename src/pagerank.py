"""Moteur unifié PageRank (PR) et PageRank Personnalisé (PPR) en PySpark DataFrame.

Contraintes strictes respectées :
- 100 % PySpark DataFrame API (aucun appel `.rdd`, aucune bibliothèque GraphFrames).
- Même moteur paramétrable pour PR (`source_page_id=None`) et PPR (`source_page_id=int`).
- Co-partitionnement strict `HashPartitioning(page_id, K)` sur `links`, `nodes` et `ranks`
  pour obtenir 0 `Exchange` sur les jointures et 1 seul `Exchange` par itération
  (celui du `groupBy` des contributions).
- Gestion exacte des dangling nodes (masse D redistribuée selon v) et vérification
  de l'invariant `sum(r(p)) == 1.0` à chaque itération.
- Coupure de lignée par `localCheckpoint(eager=True)` par défaut (ou `checkpoint(eager=True)`
  en mode repli sur HDFS/GCS).
"""

from __future__ import annotations

import contextlib
import io
import re
import time
from dataclasses import dataclass
from typing import Optional

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import (
    COL_DST,
    COL_IS_DANGLING,
    COL_OUT_DEGREE,
    COL_PAGE_ID,
    COL_RANK,
    COL_SRC,
    COL_TELEPORT,
    DAMPING_FACTOR,
    NUM_ITERATIONS,
    RANK_SUM_TOLERANCE,
)


@dataclass
class PageRankResult:
    """Résultat complet d'une exécution PageRank ou PPR."""

    ranks: DataFrame
    mode: str
    source_page_id: Optional[int]
    num_nodes: int
    num_edges: int
    num_dangling: int
    iteration_times: list[float]
    rank_sums: list[float]
    dangling_masses: list[float]
    explain_plan: str
    exchange_count: int


def capture_explain_formatted(df: DataFrame) -> str:
    """Capture la sortie texte de `df.explain(mode='formatted')` sans toucher à l'API privée."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(mode="formatted")
    return buffer.getvalue()


def count_exchanges_in_plan(explain_plan: str) -> int:
    """Compte les `Exchange` réellement exécutés dans l'arbre physique du plan formaté.

    Dans `explain(mode='formatted')` :
    - L'arbre physique apparaît sous `== Physical Plan ==`, avant les blocs détaillés `(1) ...`.
    - Les sous-arbres imbriqués sous `InMemoryRelation` représentent le plan historique qui a
      construit les DataFrames déjà mis en cache (`links_df`, `base_nodes_df`) : ils ne sont
      PAS ré-exécutés pendant l'itération (seul `InMemoryTableScan` est lu depuis le cache).
    On exclut donc ces sous-arbres déjà matérialisés pour ne compter que les `Exchange` actifs.
    """
    active_lines: list[str] = []
    in_tree = False
    skip_indent: Optional[int] = None

    for line in explain_plan.splitlines():
        stripped = line.strip()
        if stripped.startswith("== Physical Plan =="):
            in_tree = True
            continue
        if not in_tree:
            continue
        # Les sections détaillées commencent par "(1) ..." après l'arbre physique
        if re.match(r"^\(\d+\)\s+", stripped):
            break
        if not stripped:
            continue

        # Position du marqueur de branche (`+-` ou `:-`) pour connaître la profondeur dans l'arbre
        branch_match = re.search(r"(\+-|:-)", line)
        branch_col = branch_match.start() if branch_match else 0

        if skip_indent is not None:
            if branch_col > skip_indent:
                # Toujours à l'intérieur du plan historique de InMemoryRelation (non exécuté)
                continue
            skip_indent = None

        if "InMemoryRelation" in line:
            skip_indent = branch_col
            continue

        active_lines.append(line)

    return sum(1 for line in active_lines if re.search(r"\bExchange\b", line))


def prepare_graph_tables(
    edges_df: DataFrame,
    num_partitions: int,
) -> tuple[DataFrame, DataFrame, int, int, int]:
    """Prépare les tables co-partitionnées `links` et `base_nodes` à partir des arêtes `(src, dst)`.

    Les deux tables sont hash-partitionnées sur `page_id` en exactement `num_partitions`
    partitions et mises en cache (`MEMORY_AND_DISK`) une seule fois avant les itérations.
    Elles peuvent être réutilisées telles quelles pour PR puis pour PPR.

    Returns:
        Tuple `(links_df, base_nodes_df, num_nodes, num_edges, num_dangling)` :
        - `links_df` : colonnes `(page_id, dst, out_degree)` partitionnées par `page_id` (K partitions)
        - `base_nodes_df` : colonnes `(page_id, is_dangling)` partitionnées par `page_id` (K partitions)
    """
    clean_edges = edges_df.select(
        F.col(COL_SRC).cast("int").alias(COL_SRC),
        F.col(COL_DST).cast("int").alias(COL_DST),
    )

    # 1. Degré sortant deg(u) de chaque sommet ayant au moins un lien sortant
    out_degrees = clean_edges.groupBy(COL_SRC).agg(
        F.count(COL_DST).cast("int").alias(COL_OUT_DEGREE)
    )

    # 2. Table des liens enrichie de deg(u) et hash-partitionnée sur la clé de jointure `page_id` (= src)
    links_df = (
        clean_edges.join(out_degrees, on=COL_SRC, how="inner")
        .select(
            F.col(COL_SRC).alias(COL_PAGE_ID),
            F.col(COL_DST),
            F.col(COL_OUT_DEGREE),
        )
        .repartition(num_partitions, COL_PAGE_ID)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # 3. Ensemble complet des sommets V = sources U cibles (pour ne perdre aucun dangling node)
    all_nodes = (
        clean_edges.select(F.col(COL_SRC).alias(COL_PAGE_ID))
        .union(clean_edges.select(F.col(COL_DST).alias(COL_PAGE_ID)))
        .distinct()
    )

    # 4. Table de référence des sommets avec indicateur `is_dangling` (deg == 0),
    #    co-partitionnée exactement comme `links_df` sur `page_id` (K partitions)
    base_nodes_df = (
        all_nodes.join(
            out_degrees,
            all_nodes[COL_PAGE_ID] == out_degrees[COL_SRC],
            how="left",
        )
        .select(
            all_nodes[COL_PAGE_ID],
            F.col(COL_OUT_DEGREE).isNull().alias(COL_IS_DANGLING),
        )
        .repartition(num_partitions, COL_PAGE_ID)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # Matérialisation des deux tables en cache et calcul des statistiques de graphe
    num_edges = links_df.count()
    node_stats = base_nodes_df.agg(
        F.count(F.lit(1)).alias("num_nodes"),
        F.sum(F.when(F.col(COL_IS_DANGLING), F.lit(1)).otherwise(F.lit(0))).alias("num_dangling"),
    ).first()

    num_nodes = int(node_stats["num_nodes"])
    num_dangling = int(node_stats["num_dangling"] or 0)

    return links_df, base_nodes_df, num_nodes, num_edges, num_dangling


def _checkpoint_df(df: DataFrame, checkpoint_mode: str) -> DataFrame:
    """Coupe la lignée du DataFrame selon le mode choisi (`local` ou `reliable`)."""
    if checkpoint_mode == "local":
        return df.localCheckpoint(eager=True)
    if checkpoint_mode == "reliable":
        return df.checkpoint(eager=True)
    raise ValueError(
        f"Mode de checkpoint invalide : {checkpoint_mode!r}. Attendu : 'local' ou 'reliable'."
    )


def run_pagerank(
    links_df: DataFrame,
    base_nodes_df: DataFrame,
    num_nodes: int,
    num_edges: int = 0,
    num_dangling: int = 0,
    source_page_id: Optional[int] = None,
    damping_factor: float = DAMPING_FACTOR,
    num_iterations: int = NUM_ITERATIONS,
    checkpoint_mode: str = "local",
    enforce_single_exchange: bool = True,
) -> PageRankResult:
    """Exécute le PageRank classique (PR) ou personnalisé (PPR) en PySpark DataFrame.

    Formule appliquée à chaque itération :
        r(p) <- (1 - d + d * D) * v(p) + d * sum_{u -> p} (r(u) / deg(u))

    Args:
        links_df: Table `(page_id, dst, out_degree)` déjà partitionnée par `page_id` (K partitions).
        base_nodes_df: Table `(page_id, is_dangling)` déjà partitionnée par `page_id` (K partitions).
        num_nodes: Nombre total de sommets N = |V|.
        num_edges: Nombre total d'arêtes (pour traçabilité).
        num_dangling: Nombre de dangling nodes (pour traçabilité).
        source_page_id: `None` pour PR classique (v = 1/N) ; entier pour PPR (v(source) = 1).
        damping_factor: Facteur d'amortissement d (défaut 0.85).
        num_iterations: Nombre exact d'itérations (défaut 20).
        checkpoint_mode: `'local'` (`localCheckpoint(eager=True)`) ou `'reliable'` (`checkpoint(eager=True)`).
        enforce_single_exchange: Si True, lève une erreur si le plan d'une itération ne contient
            pas exactement 1 `Exchange`.

    Returns:
        `PageRankResult` contenant le DataFrame final `(page_id, rank)` et toutes les métriques.
    """
    if num_nodes <= 0:
        raise ValueError("Le graphe doit contenir au moins un sommet (num_nodes > 0).")
    if num_iterations <= 0:
        raise ValueError("num_iterations doit être strictement positif.")

    mode = "pr" if source_page_id is None else "ppr"

    # Construction du vecteur de téléportation v(p) sans casser le HashPartitioning de base_nodes_df
    if source_page_id is None:
        v_expr = F.lit(1.0 / float(num_nodes))
    else:
        source_exists = (
            base_nodes_df.filter(F.col(COL_PAGE_ID) == F.lit(int(source_page_id)))
            .limit(1)
            .count()
            > 0
        )
        if not source_exists:
            raise ValueError(
                f"La page source PPR ({source_page_id}) est absente du graphe."
            )
        v_expr = F.when(
            F.col(COL_PAGE_ID) == F.lit(int(source_page_id)), F.lit(1.0)
        ).otherwise(F.lit(0.0))

    # Table de référence des nœuds avec v(p), conservant HashPartitioning(page_id, K)
    nodes_with_v = base_nodes_df.withColumn(COL_TELEPORT, v_expr)

    # Initialisation : r^(0) = v, matérialisée et coupée pour partir sur un plan propre
    ranks = _checkpoint_df(
        nodes_with_v.withColumn(COL_RANK, F.col(COL_TELEPORT)),
        checkpoint_mode=checkpoint_mode,
    )

    iteration_times: list[float] = []
    rank_sums: list[float] = []
    dangling_masses: list[float] = []
    explain_plan = ""
    exchange_count = -1

    # On capture le plan à l'itération 2 (si >= 2 itérations) pour prouver que
    # `localCheckpoint` a bien préservé le HashPartitioning d'une itération à l'autre.
    explain_capture_iter = 2 if num_iterations >= 2 else 1

    # Agrégation initiale (une passe) : vérifie sum(r^(0)) == 1 et récupère D^(0)
    init_stats = ranks.agg(
        F.sum(COL_RANK).alias("rank_sum"),
        F.sum(
            F.when(F.col(COL_IS_DANGLING), F.col(COL_RANK)).otherwise(F.lit(0.0))
        ).alias("dangling_mass"),
    ).first()
    init_sum = float(init_stats["rank_sum"] or 0.0)
    dangling_mass = float(init_stats["dangling_mass"] or 0.0)
    if abs(init_sum - 1.0) > RANK_SUM_TOLERANCE:
        raise RuntimeError(
            f"Invariant violé à l'initialisation : "
            f"sum(r^(0)) = {init_sum:.10f} (attendu 1.0 ± {RANK_SUM_TOLERANCE})."
        )

    for iteration in range(1, num_iterations + 1):
        t0 = time.perf_counter()
        dangling_masses.append(dangling_mass)

        # 1. Jointure co-partitionnée links ⋈ ranks sur `page_id` -> 0 Exchange
        #    puis groupBy(dst) -> L'UNIQUE Exchange de l'itération (sortie HashPartitioning(page_id, K))
        contributions = (
            links_df.join(
                ranks.select(COL_PAGE_ID, COL_RANK),
                on=COL_PAGE_ID,
                how="inner",
            )
            .select(
                F.col(COL_DST).alias(COL_PAGE_ID),
                (F.col(COL_RANK) / F.col(COL_OUT_DEGREE)).alias("contrib"),
            )
            .groupBy(COL_PAGE_ID)
            .agg(F.sum("contrib").alias("incoming_rank"))
        )

        # 2. Jointure co-partitionnée nodes_with_v ⋈_left contributions sur `page_id` -> 0 Exchange
        #    (les pages sans lien entrant reçoivent incoming_rank = 0.0 via coalesce)
        teleport_factor = (1.0 - damping_factor) + damping_factor * dangling_mass
        next_ranks = nodes_with_v.join(
            contributions,
            on=COL_PAGE_ID,
            how="left_outer",
        ).select(
            F.col(COL_PAGE_ID),
            F.col(COL_TELEPORT),
            F.col(COL_IS_DANGLING),
            (
                F.lit(teleport_factor) * F.col(COL_TELEPORT)
                + F.lit(damping_factor)
                * F.coalesce(F.col("incoming_rank"), F.lit(0.0))
            ).alias(COL_RANK),
        )

        # 3. Capture et audit automatique du plan d'exécution
        if iteration == explain_capture_iter:
            explain_plan = capture_explain_formatted(next_ranks)
            exchange_count = count_exchanges_in_plan(explain_plan)
            if enforce_single_exchange and exchange_count != 1:
                raise RuntimeError(
                    f"Plan d'exécution non conforme à l'itération {iteration} : "
                    f"{exchange_count} Exchange détecté(s) au lieu de 1.\n\n{explain_plan}"
                )

        # 4. Coupure de lignée + matérialisation immédiate tout en préservant HashPartitioning(page_id, K)
        prev_ranks = ranks
        ranks = _checkpoint_df(next_ranks, checkpoint_mode=checkpoint_mode)
        prev_ranks.unpersist(blocking=False)

        # 5. En UNE seule agrégation scalaire, vérifie sum(r) == 1.0 et prépare D pour l'itération suivante
        post_stats = ranks.agg(
            F.sum(COL_RANK).alias("rank_sum"),
            F.sum(
                F.when(F.col(COL_IS_DANGLING), F.col(COL_RANK)).otherwise(F.lit(0.0))
            ).alias("dangling_mass"),
        ).first()
        post_sum = float(post_stats["rank_sum"] or 0.0)
        dangling_mass = float(post_stats["dangling_mass"] or 0.0)
        if abs(post_sum - 1.0) > RANK_SUM_TOLERANCE:
            raise RuntimeError(
                f"Invariant violé après l'itération {iteration} : "
                f"sum(r) = {post_sum:.10f} (attendu 1.0 ± {RANK_SUM_TOLERANCE})."
            )

        elapsed = time.perf_counter() - t0
        iteration_times.append(elapsed)
        rank_sums.append(post_sum)

    final_ranks = ranks.select(COL_PAGE_ID, COL_RANK)

    return PageRankResult(
        ranks=final_ranks,
        mode=mode,
        source_page_id=source_page_id,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_dangling=num_dangling,
        iteration_times=iteration_times,
        rank_sums=rank_sums,
        dangling_masses=dangling_masses,
        explain_plan=explain_plan,
        exchange_count=exchange_count,
    )
