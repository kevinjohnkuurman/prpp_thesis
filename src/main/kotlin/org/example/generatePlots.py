# python3 ./src/main/kotlin/org/example/generatePlots.py ../results/06/4
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
import re
from matplotlib.widgets import Slider
from multiprocessing import Pool
from tqdm import tqdm
from os.path import isfile, join

WITH_LINES = False
WITH_LABELS = False
WITH_TICKS = True
CARE_ABOUT_ASPECT = False


def generate_figure(data, index):
    folder = str(sys.argv[1])
    board = np.array(data["board"])
    plt.cla()
    if WITH_LINES:
        plt.pcolormesh(board.transpose(), edgecolors='k', linewidth=2)
    else:
        plt.pcolormesh(board.transpose())
    if CARE_ABOUT_ASPECT:
        plt.axis('scaled')
    plt.xlim([0, board.shape[0]])
    plt.ylim([0, board.shape[1]])

    if not WITH_TICKS:
        plt.xticks([])
        plt.yticks([])

    if WITH_LABELS:
        for placement in data['tiles']:
            plt.text(
                placement['coord']['X'] + 0.5,
                placement['coord']['Y'] + placement['tile']['Y'] - 0.5,
                f"{placement['tile']['id']}\n{'T' if placement['tile']['rotated'] else 'F' }",
                bbox=dict(facecolor='white', edgecolor='black'),
                horizontalalignment='center',
                verticalalignment='center',
                )

    plt.savefig(f"{folder}/solution{index}.png")


def generate_figure_star(args):
    return generate_figure(*args)


if __name__ == '__main__':
    folder = str(sys.argv[1])
    solverResults = None
    with open(join(folder, 'finished.json'), 'r') as f:
        solverResults = json.load(f)

    solutions = solverResults['foundSolutions']

    inputs = list(zip(solutions, [x for x in range(len(solutions))]))
    with Pool(4) as pool:
        list(tqdm(pool.imap_unordered(generate_figure_star, inputs, chunksize=1), total=len(inputs)))
