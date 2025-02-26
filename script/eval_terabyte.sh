#!/bin/bash
echo " -----  run pytorch dlrm eval  -----"

blockType="mlp"
botShape="13-512-256-64"
topShape="512-512-256-1"

# blockType="transformer"
# topShape="512-512-256-64-1"
# botShape="13-512-256-64"
# topShape="256-256-64-1"
# botShape="13-256-32"
# topShape="64-64-16-1"
# botShape="13-64-8"

sparseFeatureSize=${botShape##*-}
saveModelDir="/data/hyou37/yipin/program/dlrm/ckpt/terabyte/interaction_transformer"

# ========= device & log ========= #
if [[ $# == 1 ]]; then
    gpuUsed=$1
else
    gpuUsed="0"
fi

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
    --round-targets=True \
    --inference-only \
    --test-mini-batch-size=16384 \
    --test-num-workers=64 \
    --max-ind-range=10000000 \
    --load-model=${saveModelDir}/${blockType}_bot-${botShape}_top-${topShape}_noSparse.pth \
    --use-gpu \
    --throughput \
    --memory-map 2>&1 | tee ${saveModelDir}/eval.log

echo "done"

    # --processed-data-file=/data/hyou37/yipin/dataset/Criteo_Research/kaggleAdDisplayChallenge_processed.npz \