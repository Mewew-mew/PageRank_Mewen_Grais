#!/usr/bin/env bash
# =============================================================================
# Téléchargement streaming des graphes WikiLinkGraphs (2018 principal, 2008 bonus)
# directement dans Google Cloud Storage (0 octet de disque consommé dans Cloud Shell).
#
# Usage :
#   ./scripts/download_data.sh <GCS_BUCKET> [2018|2008|all]
# Exemple :
#   ./scripts/download_data.sh mon-bucket-pagerank 2018
# =============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <GCS_BUCKET> [2018|2008|all]" >&2
  exit 1
fi

BUCKET="$1"
YEAR_MODE="${2:-2018}"

download_year() {
  local year="$1"
  local filename="enwiki.wikilink_graph.${year}-03-01.csv.gz"
  local url="https://zenodo.org/records/2539424/files/${filename}?download=1"
  local gcs_dest="gs://${BUCKET}/raw/${filename}"

  echo ">>> Téléchargement streaming ${year} -> ${gcs_dest}"
  curl -fSL "${url}" | gcloud storage cp - "${gcs_dest}"
  echo ">>> Terminé : ${gcs_dest}"
  gcloud storage ls -lh "${gcs_dest}"
}

case "${YEAR_MODE}" in
  2018)
    download_year 2018
    ;;
  2008)
    download_year 2008
    ;;
  all)
    download_year 2018
    download_year 2008
    ;;
  *)
    echo "Erreur : année invalide '${YEAR_MODE}'. Valeurs autorisées : 2018, 2008, all." >&2
    exit 1
    ;;
esac
