#!/bin/bash
echo " -----  run pytorch dlrm train  -----"

# ========= train parameters ========= #
nbatches=2399
lrNumWarmupSteps=$((30*nbatches))
lrDecayStartStep=$((150*nbatches))
lrNumDecaySteps=$((50*nbatches))

blockType="mlp"
botShape="13-512-256-64"
topShape="512-512-256-1"

# blockType="transformer"
# topShape="512-256-1"
# botShape="13-512-128-16"

sparseFeatureSize=${botShape##*-}
saveModelDir="/data/hyou37/yipin/program/dlrm/ckpt/terabyte/mlp"

# ========= device & log ========= #
if [[ $# == 1 ]]; then
    gpuUsed=$1
else
    gpuUsed="0"
fi
timeNow=$(date +%Y-%m-%d_%H:%M_gpu${gpuUsed})

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=4 dlrm_s_pytorch.py \
CUDA_VISIBLE_DEVICES=${gpuUsed} python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=${sparseFeatureSize} \
    --block-type=${blockType} \
    --arch-mlp-bot=${botShape} \
    --arch-mlp-top=${topShape} \
    --arch-transformer-bot=${botShape} \
    --arch-transformer-top=${topShape} \
    --data-generation=dataset \
    --data-set=terabyte \
    --raw-data-file=/data/hyou37/yipin/dataset/Criteo_Terabyte/day \
    --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Terabyte/.npz \
    --loss-function='bce' \
    --round-targets=True \
    --optimizer='sgd' \
    --learning-rate=0.02 \
    --weight-decay=0.0 \
    --momentum=0.0 \
    --mini-batch-size=4096 \
    --nepochs=250 \
    --test-freq=1 \
    --print-freq=512 \
    --print-time \
    --test-mini-batch-size=16384 \
    --max-ind-range=10000000 \
    --save-model=${saveModelDir}/${blockType}_bot-${botShape}_top-${topShape}.pth \
    --use-gpu \
    --dataset-multiprocessing


    # --num-workers=64 \
    # --test-num-workers=64 \
    # --dataset-multiprocessing  #  The Terabyte dataset can be multiprocessed in an environment with more than 24 CPU cores and at least 1 TB of memory
    # 2>&1 | tee ${saveModelDir}/${timeNow}.log

    # --lr-num-warmup-steps=${lrNumWarmupSteps} \
    # --lr-decay-start-step=${lrDecayStartStep} \
    # --lr-num-decay-steps=${lrNumDecaySteps} \
    # --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_train_processed.npz \
echo "done"

# nohup ./script/train_terabyte.sh 0,1,2,3,4,5,6,7 > ./ckpt/terabyte/mlp/07-09.log 2>&1 &