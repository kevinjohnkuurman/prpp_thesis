# python3 ./src/main/kotlin/org/example/generateFrequencyGraph.py
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import csv
from matplotlib.ticker import MaxNLocator


def getNumberOfComposableSupertiles(tiles):
    potentialSuperTiles = {}
    for ai, a in enumerate(tiles):
        for bi, b in enumerate(tiles):
            if ai == bi:
                continue
            if a['X'] == b['X'] or a['Y'] == b['X'] or a['X'] == b['Y'] or a['Y'] == b['Y']:
                id = ','.join(str(x) for x in sorted([ai, bi]))
                potentialSuperTiles[id] = potentialSuperTiles.get(id, 0) + 1
    return len(potentialSuperTiles)


def getFrequency(numOfSuperTiles, data):
    forFreq = [x for x in data if x['numberComposableOfSuperTiles'] == numOfSuperTiles]
    solvable = [x for x in forFreq if x['has_solutions']]
    if len(forFreq) == 0:
        return None
    return len(solvable) / len(forFreq)


def main():
    groupName = str(sys.argv[1])
    print("estimating hardness map " + groupName)

    csvName = "./data/" + groupName + "tiles.csv"

    with open(csvName, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [{h: x for (h, x) in zip(headers, row)} for row in reader]

    results = []
    for instance in data:
        id = instance["id"]
        has_solutions = int(instance["has_solutions"]) == 1
        recursions = int(instance["recursions"])
        tiles = json.loads(instance["tiles"])
        numberComposableOfSuperTiles = getNumberOfComposableSupertiles(tiles)

        results.append({
            'id': id,
            'has_solutions': has_solutions,
            'recursions': recursions,
            'numberComposableOfSuperTiles': numberComposableOfSuperTiles
        })

    minX = min([x['numberComposableOfSuperTiles'] for x in results])
    maxX = max([x['numberComposableOfSuperTiles'] for x in results])
    XRange = list(range(0, maxX + 2))
    frequency = [getFrequency(x, results) for x in XRange]
    points = [x for x in zip(XRange, frequency) if x[1] is not None]

    print(len(results))
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt1.scatter(
        [x[0] for x in points],
        [x[1] for x in points],
    )
    plt1.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    plt1.set_ylabel('solvable frequency')
    plt.savefig("./figures_2/" + groupName + "_frequency_figures.jpg")


if __name__ == '__main__':
    main()
