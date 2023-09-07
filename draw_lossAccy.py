import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import re

CKPT_PATH2 = '/data/hyou37/yipin/program/dlrm/ckpt/terabyte/interaction_transformer/mlp_bot-13-512-256-64_top-512-512-256-1_baseline2.log'
CKPT_PATH1 = '/data/hyou37/yipin/program/dlrm/ckpt/terabyte/mlp/mlp_bot-13-512-256-64_top-512-512-256-1_baseline3.log'

# lossMsg = np.empty(shape=[2,0])


def load_message(flieName):
    loss_pattern = re.compile(r"loss (\d+\.\d+)")
    accuracy_pattern = re.compile(r"AUC (\d+\.\d+)")

    count = 0
    with open(flieName, 'r') as f:
        fileMsg = f.read()
    loss_values = [float(match) for match in re.findall(loss_pattern, fileMsg)]
    accuracy_values = [float(match) for match in re.findall(accuracy_pattern, fileMsg)]
    
    return loss_values, accuracy_values

def draw_curve(loss_values, accuracy_values, outFilePath:str = "./plot.png"):
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots()

    # Plot loss curve on the first y-axis
    loss_step = np.arange(512, (len(loss_values) + 1)*512, 512)
    ax1.plot(loss_step, loss_values, 'r-', label='Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0.12,0.16)
    ax1.tick_params('y', colors='r')
    ax1.grid(True)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot accuracy curve on the second y-axis
    accuracy_step = np.arange(len(loss_values)/len(accuracy_values)*512, (len(loss_values) + 1)*512, (len(loss_values)/len(accuracy_values))*512)
    ax2.plot(accuracy_step, accuracy_values, 'b-', label='AUC')
    ax2.set_ylabel('AUC')
    ax2.tick_params('both', colors='b')

    # Add legend
    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines], loc='lower right')

    # Save the plot as PDF
    plt.title(CKPT_PATH.split('/')[-1][:-4])
    plt.savefig(outFilePath)


def draw_singleY(data1, data2, outFilePath:str = "./plot.png"):
    fig, ax1 = plt.subplots()

    # Plot loss curve on the first y-axis
    loss_step1 = np.arange(512, (len(data1) + 1)*512, 512)
    loss_step2 = np.arange(512, (len(data2) + 1)*512, 512)
    ax1.plot(loss_step1, data1, 'r-', label='baseline')
    ax1.plot(loss_step2, data2, 'b-', label='Trans._interaction')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('AUC')
    ax1.set_ylim(0.62,0.73)
    ax1.tick_params('y', colors='r')
    ax1.grid(True)
    # Add legend
    lines = [ax1.get_lines()[0], ax1.get_lines()[1]]
    ax1.legend(lines, [line.get_label() for line in lines], loc='lower right')

    # Save the plot as PDF
    # plt.title(CKPT_PATH.split('/')[-1][:-4])
    plt.savefig(outFilePath)

loss_values1, AUC_values1 = load_message(CKPT_PATH1)
loss_values2, AUC_values2 = load_message(CKPT_PATH2)
# draw_curve(loss_values, accuracy_values)
draw_singleY(AUC_values1, AUC_values2)