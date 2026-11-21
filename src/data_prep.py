"""Préparation des données WikiLinkGraphs : anti-goulot `.csv.gz`, Parquet & sample 10 %.

Ce module résout deux pièges majeurs du sujet :
1. Un fichier `.csv.gz` (gzip) n'est pas découpable (*non-splittable*) : Spark le lit
   obligatoirement sur un seul cœur, quelle que soit la taille du cluster. On le lit
   donc UNE SEULE FOIS avec un schéma explicite (jamais `inferSchema=True`, qui
   déclencherait 2 lectures mono-cœur !), on nettoie le graphe (déduplication,
   suppression des auto-boucles), puis on réécrit en Parquet Snappy partitionné.
2. L'échantillon de 10 % est construit comme un **sous-graphe induit** sur 10 % des
   sommets tirés aléatoirement UNION `{Nantes}` UNION son voisinage direct à 1 saut,
   puis filtrage des arcs dont source ET cible appartiennent à cet ensemble.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from src.config import (
    COL_DST,
    COL_PAGE_ID,
    COL_PAGE_TITLE,
    COL_SRC,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_FRACTION,
    NANTES_PAGE_ID,
    RAW_COL_DST_ID,
    RAW_COL_DST_TITLE,
    RAW_COL_SRC_ID,
    RAW_COL_SRC_TITLE,
)

# Schéma explicite du TSV WikiLinkGraphs (évite un second scan mono-cœur du .csv.gz)
WIKILINK_RAW_SCHEMA = StructType(
    [
        StructField(RAW_COL_SRC_ID, IntegerType(), nullable=True),
        StructField(RAW_COL_SRC_TITLE, StringType(), nullable=True),
        StructField(RAW_COL_DST_ID, IntegerType(), nullable=True),
        StructField(RAW_COL_DST_TITLE, StringType(), nullable=True),
    ]
)


@dataclass(frozen=True)
class PrepMetrics:
    """Métriques de la phase de préparation (séparée des itérations PageRank)."""

    raw_read_and_write_seconds: float
    num_edges: int
    num_titles: int
    edges_path: str
    titles_path: str


def _join_uri(base_dir: str, child: str) -> str:
    """Concatène proprement un chemin local ou un URI GCS (`gs://...`)."""
    return f"{base_dir.rstrip('/')}/{child.lstrip('/')}"


def read_raw_wikilink_tsv(spark: SparkSession, raw_tsv_path: str) -> DataFrame:
    """Lit le fichier TSV WikiLinkGraphs (`.csv.gz` ou `.tsv`) en une seule passe.

    Utilise un schéma explicite (`WIKILINK_RAW_SCHEMA`) pour éviter `inferSchema=True`.
    """
    return (
        spark.read.option("sep", "\t")
        .option("header", "true")
        .schema(WIKILINK_RAW_SCHEMA)
        .csv(raw_tsv_path)
    )


def clean_edges_and_titles(raw_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Nettoie le graphe brut en graphe simple dirigé et extrait la table `page_id -> page_title`.

    Règles appliquées :
    - Rejette les lignes dont `page_id_from` ou `page_id_to` est nul.
    - Supprime les auto-boucles (`src == dst`).
    - Déduplique les arêtes `(src, dst)` pour garantir un graphe simple cohérent avec `nx.DiGraph`.
    - Extrait la table des titres depuis l'union des sources et des cibles (pour couvrir
      aussi les *dangling nodes* qui n'apparaissent qu'en cible).
    """
    valid_raw = raw_df.filter(
        F.col(RAW_COL_SRC_ID).isNotNull() & F.col(RAW_COL_DST_ID).isNotNull()
    )

    edges_df = (
        valid_raw.select(
            F.col(RAW_COL_SRC_ID).alias(COL_SRC),
            F.col(RAW_COL_DST_ID).alias(COL_DST),
        )
        .filter(F.col(COL_SRC) != F.col(COL_DST))
        .dropDuplicates([COL_SRC, COL_DST])
    )

    src_titles = valid_raw.select(
        F.col(RAW_COL_SRC_ID).alias(COL_PAGE_ID),
        F.col(RAW_COL_SRC_TITLE).alias(COL_PAGE_TITLE),
    )
    dst_titles = valid_raw.select(
        F.col(RAW_COL_DST_ID).alias(COL_PAGE_ID),
        F.col(RAW_COL_DST_TITLE).alias(COL_PAGE_TITLE),
    )
    titles_df = src_titles.union(dst_titles).dropDuplicates([COL_PAGE_ID])

    return edges_df, titles_df


def prepare_parquet_from_raw(
    spark: SparkSession,
    raw_tsv_path: str,
    output_dir: str,
    num_partitions: int,
) -> PrepMetrics:
    """Convertit le `.csv.gz` mono-cœur en fichiers Parquet découpables et partitionnés.

    Écrit deux jeux Parquet :
    - `<output_dir>/edges.parquet` : arêtes propres `(src, dst)` partitionnées en `num_partitions` fichiers.
    - `<output_dir>/titles.parquet` : table de correspondance `(page_id, page_title)`.
    """
    t0 = time.perf_counter()

    raw_df = read_raw_wikilink_tsv(spark, raw_tsv_path)
    edges_df, titles_df = clean_edges_and_titles(raw_df)

    edges_path = _join_uri(output_dir, "edges.parquet")
    titles_path = _join_uri(output_dir, "titles.parquet")

    # Répartition en `num_partitions` fichiers Parquet Snappy (parallélise toutes les lectures futures)
    edges_df.repartition(num_partitions, COL_SRC).write.mode("overwrite").parquet(edges_path)
    titles_df.repartition(num_partitions, COL_PAGE_ID).write.mode("overwrite").parquet(titles_path)

    # Comptage depuis Parquet (lecture 100 % parallèle)
    num_edges = spark.read.parquet(edges_path).count()
    num_titles = spark.read.parquet(titles_path).count()
    elapsed = time.perf_counter() - t0

    return PrepMetrics(
        raw_read_and_write_seconds=elapsed,
        num_edges=num_edges,
        num_titles=num_titles,
        edges_path=edges_path,
        titles_path=titles_path,
    )


def build_induced_sample(
    edges_df: DataFrame,
    titles_df: DataFrame,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    anchor_page_id: int = NANTES_PAGE_ID,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[DataFrame, DataFrame]:
    """Construit un sous-graphe induit sur `sample_fraction` des sommets + Nantes et son voisinage 1-saut.

    Étapes :
    1. Calcule l'ensemble complet des sommets `V = src U dst`.
    2. Échantillonne aléatoirement `sample_fraction` (ex. 10 %) des sommets avec graine fixe.
    3. Force l'inclusion de `anchor_page_id` (Nantes) et de tous ses voisins directs
       (sortants `src == anchor` et entrants `dst == anchor`).
    4. Filtre les arcs dont `src` ET `dst` appartiennent à cet ensemble `S` (sous-graphe induit).
       Les degrés sortants `deg(u)` seront recalculés sur ce sous-graphe dans `prepare_graph_tables`.
    """
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction doit être dans (0, 1], reçu : {sample_fraction}")

    all_nodes = (
        edges_df.select(F.col(COL_SRC).alias(COL_PAGE_ID))
        .union(edges_df.select(F.col(COL_DST).alias(COL_PAGE_ID)))
        .distinct()
    )

    sampled_nodes = all_nodes.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

    # Voisinage direct à 1 saut de la page ancre (Nantes) + la page ancre elle-même
    out_neighbors = edges_df.filter(F.col(COL_SRC) == F.lit(int(anchor_page_id))).select(
        F.col(COL_DST).alias(COL_PAGE_ID)
    )
    in_neighbors = edges_df.filter(F.col(COL_DST) == F.lit(int(anchor_page_id))).select(
        F.col(COL_SRC).alias(COL_PAGE_ID)
    )
    anchor_node = all_nodes.filter(F.col(COL_PAGE_ID) == F.lit(int(anchor_page_id)))

    selected_nodes = (
        sampled_nodes.union(anchor_node)
        .union(out_neighbors)
        .union(in_neighbors)
        .distinct()
    )

    # Sous-graphe induit : on ne garde que les arcs dont les deux extrémités sont dans selected_nodes
    sampled_edges = (
        edges_df.join(
            selected_nodes.select(F.col(COL_PAGE_ID).alias(COL_SRC)),
            on=COL_SRC,
            how="left_semi",
        )
        .join(
            selected_nodes.select(F.col(COL_PAGE_ID).alias(COL_DST)),
            on=COL_DST,
            how="left_semi",
        )
    )

    sampled_titles = titles_df.join(selected_nodes, on=COL_PAGE_ID, how="left_semi")
    return sampled_edges, sampled_titles


def create_induced_sample_parquet(
    spark: SparkSession,
    prepared_dir: str,
    sample_output_dir: str,
    num_partitions: int,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    anchor_page_id: int = NANTES_PAGE_ID,
    seed: int = DEFAULT_RANDOM_SEED,
) -> PrepMetrics:
    """Crée et sauvegarde sur disque/GCS le sous-graphe induit 10 % en Parquet."""
    t0 = time.perf_counter()

    edges_df, titles_df = load_prepared_graph(spark, prepared_dir)
    sampled_edges, sampled_titles = build_induced_sample(
        edges_df=edges_df,
        titles_df=titles_df,
        sample_fraction=sample_fraction,
        anchor_page_id=anchor_page_id,
        seed=seed,
    )

    edges_path = _join_uri(sample_output_dir, "edges.parquet")
    titles_path = _join_uri(sample_output_dir, "titles.parquet")

    sampled_edges.repartition(num_partitions, COL_SRC).write.mode("overwrite").parquet(edges_path)
    sampled_titles.repartition(num_partitions, COL_PAGE_ID).write.mode("overwrite").parquet(titles_path)

    num_edges = spark.read.parquet(edges_path).count()
    num_titles = spark.read.parquet(titles_path).count()
    elapsed = time.perf_counter() - t0

    return PrepMetrics(
        raw_read_and_write_seconds=elapsed,
        num_edges=num_edges,
        num_titles=num_titles,
        edges_path=edges_path,
        titles_path=titles_path,
    )


def load_prepared_graph(spark: SparkSession, prepared_dir: str) -> tuple[DataFrame, DataFrame]:
    """Charge en parallèle les tables Parquet `(edges, titles)` déjà préparées."""
    edges_df = spark.read.parquet(_join_uri(prepared_dir, "edges.parquet"))
    titles_df = spark.read.parquet(_join_uri(prepared_dir, "titles.parquet"))
    return edges_df, titles_df
