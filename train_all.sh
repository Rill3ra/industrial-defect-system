#!/bin/bash
# train_all.sh — обучение всех моделей последовательно

set -e
echo "======================================"
echo "TRAINING ALL MODELS (30 epochs each)"
echo "======================================"

MODELS=("resnet50" "densenet121" "efficientnet_b0" "vit_b16" "mobilenet_v3")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "────────────────────────────────────"
    echo "Training: $MODEL"
    echo "────────────────────────────────────"
    python -m src.training.train_classifier --model "$MODEL" --epochs 30
done

echo ""
echo "────────────────────────────────────"
echo "Training: PatchCore (memory bank)"
echo "────────────────────────────────────"
python -m src.training.train_patchcore

echo ""
echo "======================================"
echo "ALL DONE"
echo "======================================"
