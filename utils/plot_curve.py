import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import numpy as np
import torch

def plot_result(LOSS, ACC, note):

    x = [i for i in range(len(LOSS))]
    _plot_(x, np.array(LOSS), "Loss", note)
    _plot_(x, np.array(ACC), "Acc", note)


def _plot_(x, Result, Flag, note):
    #lr = opt.lr
    #batch_size = opt.batch_size
    plt.plot(x, Result[:, 0], 'k-', label="Train %s" % Flag)
    plt.plot(x, Result[:, 1], 'r-', label="Val   %s" % Flag)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(Flag)
    plt.title(Flag + " %s" % note)
    os.makedirs("./Samples/Results", exist_ok=True)
    plt.savefig("./Samples/Results/{}-{}.png".format(Flag, note))
    plt.close()
