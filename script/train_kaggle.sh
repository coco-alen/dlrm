#!/bin/bash
echo " -----  run pytorch dlrm train  -----"

# ========= train parameters ========= #
nbatches=2399
lrNumWarmupSteps=$((30*nbatches))
lrDecayStartStep=$((150*nbatches))
lrNumDecaySteps=$((50*nbatches))
# blockType="mlp"
blockType="transformer"
botShape="13-512-256-64-16"
# topShape="512-256-1"
# botShape="13-512-128-16"
topShape="512-256-128-1"

saveModelDir="/data/hyou37/yipin/program/dlrm/ckpt/vanilla_transformer"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u dlrm_s_pytorch.py \
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
    --loss-function=bce \
    --round-targets=True \
    --optimizer='adamw' \
    --learning-rate=0.001 \
    --mini-batch-size=16384 \
    --nepochs=200 \
    --test-freq=1 \
    --print-freq=512 \
    --print-time \
    --num-workers=64 \
    --test-num-workers=64 \
    --save-model=${saveModelDir}/${blockType}_bot-${botShape}_top-${topShape}.pth \
    --use-gpu  \
    --dataset-multiprocessing

    # --lr-num-warmup-steps=${lrNumWarmupSteps} \
    # --lr-decay-start-step=${lrDecayStartStep} \
    # --lr-num-decay-steps=${lrNumDecaySteps} \

echo "done"
