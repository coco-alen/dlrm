echo " -----  run pytorch dlrm train  -----"

export MASTER_ADDR=localhost
export MASTER_PORT=5678

python -m torch.distributed.launch --nproc_per_node=8  dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/nethome/hyou37/yipin/dataset/Criteo_Research/train.txt \
    --loss-function=bce --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --save-model=/nethome/hyou37/yipin/program/dlrm/ckpt/test \
    --use-gpu 


echo "done"