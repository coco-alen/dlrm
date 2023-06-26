#!/bin/bash
echo " -----  run pytorch dlrm eval  -----"

# blockType="mlp"
blockType="transformer"
botShape="13-512-256-64-16"
# topShape="512-256-1"
# botShape="13-512-128-16"
topShape="512-256-128-1"

saveModelDir="/data/hyou37/yipin/program/dlrm/ckpt/vanilla_transformer"

CUDA_VISIBLE_DEVICES=0 python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --block-type=${blockType} \
    --arch-mlp-bot=${botShape} \
    --arch-mlp-top=${topShape} \
    --arch-transformer-bot=${botShape} \
    --arch-transformer-top=${topShape} \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/data/hyou37/yipin/dataset/Criteo_Research/train.txt \
    --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_train_processed.npz \
    --round-targets=True \
    --inference-only \
    --test-mini-batch-size=16384 \
    --test-num-workers=64 \
    --load-model=${saveModelDir}/${blockType}_bot-${botShape}_top-${topShape}.pth \
    --use-gpu 

echo "done"

    # --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_processed.npz \