#!/usr/bin/env bash
# =============================================================================
# download_metabric.sh
# Downloads METABRIC brca_metabric files from cBioPortal datahub via HTTP.
# Bypasses Git LFS — uses direct media URLs so no git-lfs install needed.
#
# Usage:
#   bash scripts/download_metabric.sh
#   bash scripts/download_metabric.sh /custom/output/path
#
# Output directory: $1 or ./data/brca_metabric/
# =============================================================================

set -euo pipefail

OUT="${1:-./data/brca_metabric}"
mkdir -p "$OUT"

BASE="https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric"

FILES=(
    "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
    "data_cna.txt"
    "data_mutations.txt"
    "data_clinical_patient.txt"
)

echo "============================================================"
echo "  Downloading METABRIC (brca_metabric) from cBioPortal"
echo "  Output: $OUT"
echo "============================================================"

for fname in "${FILES[@]}"; do
    url="$BASE/$fname"
    dest="$OUT/$fname"

    if [ -f "$dest" ]; then
        size=$(du -sh "$dest" | cut -f1)
        echo "  [SKIP] $fname already exists ($size)"
        continue
    fi

    echo -n "  Downloading $fname ... "
    if curl -fSL --retry 3 --retry-delay 5 \
            --connect-timeout 30 \
            --max-time 600 \
            -o "$dest" "$url"; then
        size=$(du -sh "$dest" | cut -f1)
        echo "done ($size)"
    else
        echo "FAILED"
        echo "  Try manually: wget '$url' -O '$dest'"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "  METABRIC download complete"
echo "  Files in: $OUT"
ls -lh "$OUT"
echo "============================================================"
echo ""
echo "In your pipeline config or CLI:"
echo "  python run_pipeline.py /path/to/tcga/ --metabric-dir $OUT"