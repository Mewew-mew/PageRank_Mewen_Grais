"""Configuration centrale et fabrique de SparkSession pour le projet PageRank.

Ce module rassemble :
1. Les constantes mathématiques et métier imposées par le sujet.
2. Les métadonnées des jeux de données WikiLinkGraphs (2018 principal, 2008 bonus).
3. La création d'une SparkSession configurée pour garantir le co-partitionnement
   (0 Exchange sur la jointure liens/rangs) et la reproductibilité des benchmarks.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

from pyspark.sql import SparkSession


# =============================================================================
# 1. Constantes mathématiques & sujet (ne pas modifier)
# =============================================================================
DAMPING_FACTOR: float = 0.85
NUM_ITERATIONS: int = 20
NANTES_PAGE_ID: int = 67453

# Tolérance pour vérifier l'invariant mathématique : sum(r(p)) == 1.0 à chaque itération
RANK_SUM_TOLERANCE: float = 1e-6

# =============================================================================
# 2. Jeu de données WikiLinkGraphs (enwiki)
# =============================================================================
WIKILINK_DATASETS: dict[int, dict[str, str]] = {
    2018: {
        "filename": "enwiki.wikilink_graph.2018-03-01.csv.gz",
        "url": "https://zenodo.org/records/2539424/files/enwiki.wikilink_graph.2018-03-01.csv.gz?download=1",
    },
    2008: {
        "filename": "enwiki.wikilink_graph.2008-03-01.csv.gz",
        "url": "https://zenodo.org/records/2539424/files/enwiki.wikilink_graph.2008-03-01.csv.gz?download=1",
    },
}

# Colonnes brutes du TSV WikiLinkGraphs
RAW_COL_SRC_ID = "page_id_from"
RAW_COL_SRC_TITLE = "page_title_from"
RAW_COL_DST_ID = "page_id_to"
RAW_COL_DST_TITLE = "page_title_to"

# Colonnes normalisées internes (schéma compact Int32 / Double)
COL_PAGE_ID = "page_id"
COL_PAGE_TITLE = "page_title"
COL_SRC = "src"
COL_DST = "dst"
COL_OUT_DEGREE = "out_degree"
COL_IS_DANGLING = "is_dangling"
COL_RANK = "rank"
COL_TELEPORT = "v"

# =============================================================================
# 3. Paramètres par défaut : échantillonnage, partitionnement et clusters GCP
# =============================================================================
DEFAULT_SAMPLE_FRACTION: float = 0.10
DEFAULT_RANDOM_SEED: int = 42

# 96 partitions est un multiple commun de 8 (2 workers x 4 vCPU),
# 16 (4 workers x 4 vCPU) et 24 (6 workers x 4 vCPU) -> équilibrage exact
# des tâches sur les 3 topologies Dataproc (n2-standard-4).
DEFAULT_NUM_PARTITIONS: int = 96
LOCAL_TEST_NUM_PARTITIONS: int = 4

# Seuil d'analyse et Top-K
DEFAULT_TOP_K: int = 20


@dataclass(frozen=True)
class SparkConfig:
    """Paramètres d'exécution Spark pour le moteur PageRank."""

    app_name: str = "Wikipedia-PageRank-PySpark"
    num_partitions: int = DEFAULT_NUM_PARTITIONS
    master: Optional[str] = None
    disable_aqe: bool = True
    disable_auto_broadcast: bool = True
    checkpoint_dir: Optional[str] = None


def build_spark_session(config: Optional[SparkConfig] = None) -> SparkSession:
    """Crée et configure une SparkSession optimisée pour le co-partitionnement.

    Points clés de configuration :
    - `spark.sql.shuffle.partitions = K` : fixe exactement le nombre de partitions
      en sortie de `groupBy`, afin qu'il reste aligné avec `links` (partitionné sur K).
    - `spark.sql.adaptive.enabled = false` : empêche l'AQE de fusionner (coalesce)
      dynamiquement les partitions après un shuffle, ce qui casserait l'alignement
      des HashPartitioning entre `ranks` et `links`.
    - `spark.sql.autoBroadcastJoinThreshold = -1` : force l'utilisation du
      SortMergeJoin co-partitionné (0 Exchange) au lieu d'un BroadcastHashJoin
      (qui ajouterait un BroadcastExchange sur les petits graphes ou le sample 10 %).
    """
    cfg = config or SparkConfig()

    # Sous Windows, PySpark cherche par défaut `python3` (alias Microsoft Store cassé)
    # si PYSPARK_PYTHON n'est pas défini. `setdefault` préserve la config Dataproc sur cluster.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = SparkSession.builder.appName(cfg.app_name)
    if cfg.master:
        builder = builder.master(cfg.master)
        if cfg.master.startswith("local"):
            # Évite les timeouts socket sous Windows en forçant l'interface loopback locale
            builder = builder.config("spark.driver.host", "127.0.0.1").config(
                "spark.driver.bindAddress", "127.0.0.1"
            )

    builder = (
        builder.config("spark.sql.shuffle.partitions", str(cfg.num_partitions))
        .config("spark.sql.adaptive.enabled", "false" if cfg.disable_aqe else "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )

    if cfg.disable_auto_broadcast:
        # -1 désactive le broadcast automatique pour garantir une jointure locale co-partitionnée
        builder = builder.config("spark.sql.autoBroadcastJoinThreshold", "-1")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if cfg.checkpoint_dir:
        spark.sparkContext.setCheckpointDir(cfg.checkpoint_dir)

    return spark
