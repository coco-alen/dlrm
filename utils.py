# others
from os import path
import time
# numpy
import numpy as np
# pytorch
import torch

import extend_distributed as ext_dist

def print_args(args):
    args = str(args)
    print(" ======== args start =========")
    message = args[args.find("(")+1:args.find(")")]
    message = message.split(",")
    message = "\n".join(message)
    print(message)
    print(" ======== args end =========\n")


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

def throughput(args,
    dlrm,
    test_ld,
    use_gpu,
    device,
    ndevices = 1,
    repeat = 100):

    testBatch = next(iter(test_ld))
    X_test, lS_o, lS_i, T_test, W_test, CBPP_test = unpack_batch(testBatch)

    # Skip the batch if batch size not multiple of total ranks
    if ext_dist.my_size > 1 and X_test.size(0) % ext_dist.my_size != 0:
        print("Warning: Skiping throughput with size %d, because batch size not multiple of total ranks" % (i, X_test.size(0)))
        return

    if use_gpu:  # .cuda()
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        if ndevices == 1:
            lS_i = (
                [S_i.to(device) for S_i in lS_i]
                if isinstance(lS_i, list)
                else lS_i.to(device)
            )
            lS_o = (
                [S_o.to(device) for S_o in lS_o]
                if isinstance(lS_o, list)
                else lS_o.to(device)
            )
    X_test = X_test.to(device)

    # throughput test
    batch_size = X_test.size(0)
    dlrm.to(device).eval()

    for i in range(repeat):
        dlrm(X_test, lS_o, lS_i)
    torch.cuda.synchronize()
    print(f"throughput averaged with {repeat} times")
    tic1 = time.time()
    for i in range(repeat):
        dlrm(X_test, lS_o, lS_i)
        torch.cuda.synchronize()
    tic2 = time.time()
    print(
        f"batch_size {batch_size} throughput {repeat * batch_size / (tic2 - tic1)}"
    )
    print(
        f"batch_size {batch_size} latency {(tic2 - tic1) / repeat * 1000} ms"
    )
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True, with_flops=True) as prof:
        dlrm(X_test, lS_o, lS_i)

    record_name = args.load_model.split("/")[-1].split(".")[0]
    prof.export_chrome_trace('./ckpt/profile/'+ record_name +'.json')
    prof.export_stacks('./ckpt/profile/'+ record_name +'_cpu_stack.json', metric="self_cpu_time_total")
    prof.export_stacks('./ckpt/profile/'+ record_name +'_gpu_stack.json', metric="self_cuda_time_total")

    # resultList = prof.table().split("\n")
    # for eachLine in resultList:
    #     if "0.000" in eachLine:
    #         continue
    #     print(eachLine)