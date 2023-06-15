#!/bin/bash
echo " -----  run pytorch dlrm eval  -----"

CUDA_VISIBLE_DEVICES=0 python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/data/hyou37/yipin/dataset/Criteo_Research/train.txt \
    --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_train_processed.npz \
    --round-targets=True \
    --inference-only \
    --test-mini-batch-size=16384 \
    --test-num-workers=64 \
    --load-model=/data/hyou37/yipin/program/dlrm/ckpt/test/MLP_bot-13-512-256-64-16_top-512-256-1.pth \
    --use-gpu 

echo "done"

    # --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_processed.npz \