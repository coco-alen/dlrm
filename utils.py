# others
from os import path
# numpy
import numpy as np
# pytorch
import torch

def print_args(args):
    args = str(args)
    print(" ======== args start =========")
    message = args[args.find("(")+1:args.find(")")]
    message = message.split(",")
    message = "\n".join(message)
    print(message)
    print(" ======== args end =========\n")

