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

def generate_figure(name):
    folder = str(sys.argv[1])
    matches = re.search(r"solution(?P<index>\d+)\.json", name)
    if matches is None:
        return
    index = matches.group('index')
    if index is None:
        return

    with open(join(folder, name), 'r') as f:
        data = json.load(f)
        board = np.array(data["board"])
        plt.cla()
        plt.pcolormesh(board.transpose(), edgecolors='k', linewidth=2)
        plt.xlim([0, board.shape[0]])
        plt.ylim([0, board.shape[1]])

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

if __name__ == '__main__':
    folder = str(sys.argv[1])
    names = list(os.listdir(folder))
    with Pool(4) as pool:
        list(tqdm(pool.imap(generate_figure, names), total=len(names)))
