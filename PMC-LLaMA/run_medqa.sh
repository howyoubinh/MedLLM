#!/bin/bash

# Default values
MODEL_PATH="axiong/PMC_LLaMA_13B"
DATASET_PATH="test_4_options.jsonl"
OUTPUT_DIR="inferenced_result_dir"
NUM_SAMPLES=10
PRECISION="float16"

# Create timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate required files
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: CUDA not found. Script may run slowly on CPU."
fi

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting MedQA inference run at $(date)" > "$LOG_FILE"

# Helper function to run inference
run_inference() {
    local aug_types="$1"
    local output_subdir="$2"
    
    echo "Running inference with augmentations: ${aug_types:-none}"
    echo "Output directory: $output_subdir"
    
    mkdir -p "$output_subdir"
    
    if [ -z "$aug_types" ]; then
        python medqa_inference.py \
            --model-name-or-path "$MODEL_PATH" \
            --dataset-name "$DATASET_PATH" \
            --write-dir "$output_subdir" \
            --num-samples "$NUM_SAMPLES" \
            --precision "$PRECISION" \
            2>&1 | tee -a "$LOG_FILE"
    else
        python medqa_inference.py \
            --model-name-or-path "$MODEL_PATH" \
            --dataset-name "$DATASET_PATH" \
            --write-dir "$output_subdir" \
            --num-samples "$NUM_SAMPLES" \
            --precision "$PRECISION" \
            --augmentations $aug_types \
            2>&1 | tee -a "$LOG_FILE"
    fi
}

# Run baseline (no augmentations)
echo "Running baseline without augmentations..."
run_inference "" "${OUTPUT_DIR}/baseline"

# Run individual augmentations
echo "Running individual augmentations..."
for aug in synonyms paraphrase shuffle expand; do
    run_inference "$aug" "${OUTPUT_DIR}/${aug}"
done

# Run combinations
echo "Running augmentation combinations..."
run_inference "synonyms paraphrase" "${OUTPUT_DIR}/synonyms_paraphrase"
run_inference "shuffle expand" "${OUTPUT_DIR}/shuffle_expand"
# run_inference "synonyms shuffle expand" "${OUTPUT_DIR}/combined_all"

# Generate summary
echo "Creating summary..."
{
    echo "Run Summary"
    echo "============"
    echo "Timestamp: $TIMESTAMP"
    echo "Model: $MODEL_PATH"
    echo "Dataset: $DATASET_PATH"
    echo "Samples: $NUM_SAMPLES"
    echo "Precision: $PRECISION"
    echo
    echo "Augmentations run:"
    echo "- Baseline (no augmentations)"
    echo "- Individual: synonyms, paraphrase, shuffle, expand"
    echo "- Combinations: synonyms+paraphrase, shuffle+expand, synonyms+shuffle+expand"
} > "${OUTPUT_DIR}/summary.txt"

echo "Run completed. Check ${OUTPUT_DIR}/summary.txt for details."