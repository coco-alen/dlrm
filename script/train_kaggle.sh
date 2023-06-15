#!/bin/bash
echo " -----  run pytorch dlrm train  -----"

nbatches=2399
lrNumWarmupSteps=$((10*nbatches))
lrDecayStartStep=$((40*nbatches))
lrNumDecaySteps=$((60*nbatches))

# python -u -m torch.distributed.launch --nproc_per_node=8  dlrm_s_pytorch.py \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/nethome/hyou37/yipin/dataset/Criteo_Research/train.txt \
    --processed-data-file=/nethome/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=16384 \
    --nepochs=70 \
    --test-freq=1 \
    --lr-num-warmup-steps=${lrNumWarmupSteps} \
    --lr-decay-start-step=${lrDecayStartStep} \
    --lr-num-decay-steps=${lrNumDecaySteps} \
    --print-freq=512 \
    --print-time \
    --num-workers=64 \
    --test-num-workers=64 \
    --save-model=/nethome/hyou37/yipin/program/dlrm/ckpt/test/MLP_bot-13-512-256-64-16_top-512-256-1.pth \
    --use-gpu  \
    --dataset-multiprocessing

echo "done"