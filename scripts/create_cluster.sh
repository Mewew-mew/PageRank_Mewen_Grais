#!/usr/bin/env bash
# =============================================================================
# Création d'un cluster Google Cloud Dataproc 2.2 (Spark 3.5.x / Python 3.11)
# calibré pour 2, 4 ou 6 workers à matériel strictement identique (<= 32 vCPU).
#
# Architecture matérielle :
#   - 1 master  : n2-standard-4 (4 vCPU, 16 Go RAM)
#   - N workers : n2-standard-4 (4 vCPU, 16 Go RAM) avec N in {2, 4, 6}
#   - Total vCPU :
#       * 2 workers -> (1 + 2) * 4 = 12 vCPU <= 32 vCPU
#       * 4 workers -> (1 + 4) * 4 = 20 vCPU <= 32 vCPU
#       * 6 workers -> (1 + 6) * 4 = 28 vCPU <= 32 vCPU
#
# Usage :
#   ./scripts/create_cluster.sh <NUM_WORKERS: 2|4|6> <REGION> [CLUSTER_PREFIX]
# Exemple :
#   ./scripts/create_cluster.sh 4 europe-west1 pagerank-cluster
# =============================================================================

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <NUM_WORKERS: 2|4|6> <REGION> [CLUSTER_PREFIX]" >&2
  exit 1
fi

NUM_WORKERS="$1"
REGION="$2"
PREFIX="${3:-pagerank-cluster}"
MACHINE_TYPE="n2-standard-4"
IMAGE_VERSION="2.2-debian12"

if [[ "${NUM_WORKERS}" != "2" && "${NUM_WORKERS}" != "4" && "${NUM_WORKERS}" != "6" ]]; then
  echo "Erreur : NUM_WORKERS doit être 2, 4 ou 6 (reçu : ${NUM_WORKERS})." >&2
  exit 1
fi

TOTAL_VCPU=$(( (1 + NUM_WORKERS) * 4 ))
if (( TOTAL_VCPU > 32 )); then
  echo "Erreur : dépassement du quota 32 vCPU (${TOTAL_VCPU} vCPU)." >&2
  exit 1
fi

CLUSTER_NAME="${PREFIX}-${NUM_WORKERS}w"

echo ">>> Création du cluster Dataproc '${CLUSTER_NAME}'"
echo "    Région : ${REGION} | Image : ${IMAGE_VERSION}"
echo "    Matériel : 1 master + ${NUM_WORKERS} workers en ${MACHINE_TYPE} (${TOTAL_VCPU} vCPU / 32 max)"

gcloud dataproc clusters create "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --image-version="${IMAGE_VERSION}" \
  --master-machine-type="${MACHINE_TYPE}" \
  --master-boot-disk-size=50GB \
  --master-boot-disk-type=pd-balanced \
  --num-workers="${NUM_WORKERS}" \
  --worker-machine-type="${MACHINE_TYPE}" \
  --worker-boot-disk-size=50GB \
  --worker-boot-disk-type=pd-balanced \
  --max-idle=30m \
  --properties="spark:spark.sql.adaptive.enabled=false,spark:spark.sql.autoBroadcastJoinThreshold=-1,spark:spark.serializer=org.apache.spark.serializer.KryoSerializer"

echo ""
echo ">>> Cluster '${CLUSTER_NAME}' prêt !"
echo ">>> RAPPEL COÛTS : pense à supprimer le cluster dès la fin des mesures :"
echo "    gcloud dataproc clusters delete ${CLUSTER_NAME} --region=${REGION} --quiet"
