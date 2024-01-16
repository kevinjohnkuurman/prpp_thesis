# python3 ./src/main/kotlin/org/example/generateAdjacencyGraphs.py 06 true
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


def main():
    groupName = str(sys.argv[1])
    print("estimating hardness map " + groupName)

    resultsFolder = "../results/" + groupName
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

        hardnessMeasure = numberComposableOfSuperTiles

        results.append({
            'has_solutions': has_solutions,
            'recursions': recursions,
            'hardnessMeasure': hardnessMeasure
        })

    print(len(results))
    plt.rcParams.update({'font.size': 14})
    C_SOLVABLE = np.array([0, 140, 0]) / 255.0
    C_UNSOLVABLE = np.array([255, 0, 0]) / 255.0

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt1.scatter(
        [x['hardnessMeasure'] for x in results],
        [x['recursions'] for x in results],
        alpha=0.1,
        c=[C_SOLVABLE if x['has_solutions'] else C_UNSOLVABLE for x in results]
    )
    plt1.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    plt1.set_ylabel('hardness')
    plt.savefig("./figures/" + groupName + "_same_sided_edges_hardness.jpg")
    # plt.show()

if __name__ == '__main__':
    main()
