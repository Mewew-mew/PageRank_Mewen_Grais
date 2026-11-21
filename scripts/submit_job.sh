#!/usr/bin/env bash
# =============================================================================
# Soumission d'un job PySpark sur Google Cloud Dataproc (`prepare`, `sample`, `run`).
#
# Empaquette automatiquement `src/` en `src.zip` (`--py-files`) et règle la mémoire
# et le partitionnement identiquement sur les clusters 2, 4 et 6 workers `n2-standard-4`.
#
# Usage :
#   ./scripts/submit_job.sh <CLUSTER_NAME> <REGION> <STAGE: prepare|sample|run> <ARGS_MAIN...>
#
# Exemples :
#   # 1. Conversion .csv.gz -> Parquet (à faire 1 seule fois !)
#   ./scripts/submit_job.sh pagerank-cluster-4w europe-west1 prepare \
#       --raw-input gs://BUCKET/raw/enwiki.wikilink_graph.2018-03-01.csv.gz \
#       --prepared-dir gs://BUCKET/prepared/2018_full --num-partitions 96
#
#   # 2. Création du sous-graphe induit 10 % centré sur Nantes
#   ./scripts/submit_job.sh pagerank-cluster-4w europe-west1 sample \
#       --prepared-dir gs://BUCKET/prepared/2018_full \
#       --sample-dir gs://BUCKET/prepared/2018_sample10 --num-partitions 96
#
#   # 3. Exécution PR + PPR + benchmarks + plan explain()
#   ./scripts/submit_job.sh pagerank-cluster-4w europe-west1 run \
#       --prepared-dir gs://BUCKET/prepared/2018_full \
#       --results-dir gs://BUCKET/results/full_2018_4w \
#       --cluster-nodes 4 --num-partitions 96 --dataset-label full_2018
# =============================================================================

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <CLUSTER_NAME> <REGION> <STAGE: prepare|sample|run> [ARGS...]" >&2
  exit 1
fi

CLUSTER_NAME="$1"
REGION="$2"
STAGE="$3"
shift 3

# Résolution de la racine du dépôt (indépendante du dossier courant)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Empaquetage propre du package `src/` pour distribution aux exécuteurs Dataproc
ZIP_PATH="/tmp/pagerank_src.zip"
rm -f "${ZIP_PATH}"
(cd "${REPO_ROOT}" && zip -rq "${ZIP_PATH}" src -x "*/__pycache__/*" "*.pyc")

# Propriétés Spark calibrées pour n2-standard-4 (4 vCPU / 16 Go RAM par worker) :
# - 1 exécuteur par worker (4 cœurs, 10 Go heap + 2 Go overhead = 12 Go <= 16 Go)
# - K = 96 partitions fixes (multiple commun de 8, 16 et 24 cœurs)
# - AQE = false et autoBroadcastJoinThreshold = -1 (garantie 1 seul Exchange)
SPARK_PROPS=(
  "spark.executor.cores=4"
  "spark.executor.memory=10g"
  "spark.executor.memoryOverhead=2g"
  "spark.driver.memory=8g"
  "spark.sql.shuffle.partitions=96"
  "spark.sql.adaptive.enabled=false"
  "spark.sql.autoBroadcastJoinThreshold=-1"
  "spark.serializer=org.apache.spark.serializer.KryoSerializer"
  "spark.dynamicAllocation.enabled=false"
)
PROPS_CSV="$(IFS=,; echo "${SPARK_PROPS[*]}")"

echo ">>> Soumission du stage '${STAGE}' sur ${CLUSTER_NAME} (${REGION})..."
gcloud dataproc jobs submit pyspark "${REPO_ROOT}/src/main.py" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --py-files="${ZIP_PATH}" \
  --properties="${PROPS_CSV}" \
  -- "${STAGE}" "$@"
