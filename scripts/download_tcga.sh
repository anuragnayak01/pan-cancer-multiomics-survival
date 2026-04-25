 
set -euo pipefail

OUT="${1:-./data/tcga}"
mkdir -p "$OUT"

echo "============================================================"
echo "  TCGA Pan-Cancer Multi-Omics Data"
echo "  Output: $OUT"
echo "============================================================"

# ── Option 1: gdown from Google Drive ────────────────────────────────────────
# If you have shared the parquet files on Google Drive, paste the IDs here.
# Format: gdown --id <GOOGLE_DRIVE_FILE_ID> -O <output_file>

if command -v gdown &> /dev/null; then
    echo "  gdown detected. Using Google Drive download."
    echo ""
    echo "  NOTE: Replace GOOGLE_DRIVE_FILE_ID with your actual file IDs."
    echo "  You can get these from the shareable link of each Drive file."
    echo ""

    declare -A DRIVE_IDS=(
        ["mRNA.parquet"]="YOUR_MRNA_DRIVE_ID"
        ["miRNA.parquet"]="YOUR_MIRNA_DRIVE_ID"
        ["CNV.parquet"]="YOUR_CNV_DRIVE_ID"
        ["mutations.parquet"]="YOUR_MUTATIONS_DRIVE_ID"
        ["clinical.parquet"]="YOUR_CLINICAL_DRIVE_ID"
    )

    for fname in "${!DRIVE_IDS[@]}"; do
        file_id="${DRIVE_IDS[$fname]}"
        dest="$OUT/$fname"

        if [ -f "$dest" ]; then
            size=$(du -sh "$dest" | cut -f1)
            echo "  [SKIP] $fname already exists ($size)"
            continue
        fi

        if [ "$file_id" = "YOUR_${fname%%.*}_DRIVE_ID" ]; then
            echo "  [SKIP] $fname — no Drive ID configured (edit this script)"
            continue
        fi

        echo -n "  Downloading $fname from Google Drive ... "
        gdown --id "$file_id" -O "$dest" --quiet
        size=$(du -sh "$dest" | cut -f1)
        echo "done ($size)"
    done

else
    echo "  gdown not found. Installing..."
    pip install -q gdown
    echo ""
    echo "  Re-run this script after installation."
    echo "    bash scripts/download_tcga.sh $OUT"
    exit 0
fi

# ── Option 2: Manual GDC download instructions ───────────────────────────────
echo ""
echo "============================================================"
echo "  If Drive download failed, download from GDC manually:"
echo ""
echo "  1. Go to: https://portal.gdc.cancer.gov"
echo "  2. Filter: Program = TCGA, Data Type = Gene Expression Quantification"
echo "  3. Download manifest → use gdc-client:"
echo "       pip install gdc-client"
echo "       gdc-client download -m manifest.txt -d $OUT"
echo ""
echo "  Required data types:"
echo "    mRNA:      Gene Expression Quantification (STAR - Counts)"
echo "    miRNA:     miRNA Expression Quantification"
echo "    CNV:       Copy Number Segment (Masked)"
echo "    Mutation:  Masked Somatic Mutation (MuSE)"
echo "    Clinical:  Clinical Supplement"
echo ""
echo "  After download, convert to parquet using:"
echo "    python -c \"import pandas as pd; ..."
echo "============================================================"

echo ""
echo "Files in $OUT:"
ls -lh "$OUT" 2>/dev/null || echo "  (empty)"