import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import re

CKPT_PATH = '/data/hyou37/yipin/program/dlrm/ckpt/vanilla_transformer/transformer_bot-13-256-128-64-32_top-256-128-64-1_wbce.log'

# lossMsg = np.empty(shape=[2,0])


def load_message(flieName):
    loss_pattern = re.compile(r"loss (\d+\.\d+)")
    accuracy_pattern = re.compile(r"accuracy (\d+\.\d+)")

    count = 0
    with open(flieName, 'r') as f:
        fileMsg = f.read()
    loss_values = [float(match) for match in re.findall(loss_pattern, fileMsg)]
    accuracy_values = [float(match) for match in re.findall(accuracy_pattern, fileMsg)]
    
    return loss_values, accuracy_values

def draw_curve(loss_values, accuracy_values, outFilePath:str = "./plot.pdf"):
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots()

    # Plot loss curve on the first y-axis
    loss_step = np.arange(1, len(loss_values) + 1)
    ax1.plot(loss_step, loss_values, 'r-', label='Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.tick_params('y', colors='r')
    ax1.grid(True)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot accuracy curve on the second y-axis
    accuracy_step = np.arange(len(loss_values)/len(accuracy_values), len(loss_values) + 1, len(loss_values)/len(accuracy_values))
    ax2.plot(accuracy_step, accuracy_values, 'b-', label='Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.tick_params('both', colors='b')

    # Add legend
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines], loc='lower right')

    # Save the plot as PDF
    plt.title(CKPT_PATH.split('/')[-1][:-4])
    plt.savefig(outFilePath)

loss_values, accuracy_values = load_message(CKPT_PATH)
draw_curve(loss_values, accuracy_values)