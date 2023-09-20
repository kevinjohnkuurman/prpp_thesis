# python3 ./src/main/kotlin/org/example/generateAdjacencyGraphs.py 06 true
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
import re
import itertools
from multiprocessing import Pool
from tqdm import tqdm
from os.path import isfile, join

def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def subPathToString(x):
    if len(x) == 1:
        return x[0]
    return "[" + str(",".join([subPathToString(s) for s in x])) + "]"

def getPlacedTileEdges(placedTile):
    coord = placedTile['coord']
    tile = placedTile['tile']
    id = tile['id']
    rotated = tile['rotated']
    tx = f'X[{id}]'
    ty = f'Y[{id}]'
    if rotated:
        tx, ty = ty, tx

    c1 = (coord['X'],               coord['Y'])
    c2 = (coord['X'],               coord['Y'] + tile['Y'])
    c3 = (coord['X'] + tile['X'],   coord['Y'])
    c4 = (coord['X'] + tile['X'],   coord['Y'] + tile['Y'])

    return [
        ('X', tx, rotated, id, c1, c3),
        ('X', tx, rotated, id, c2, c4),
        ('Y', ty, rotated, id, c1, c2),
        ('Y', ty, rotated, id, c3, c4),
    ]

def fileIsSolution(name):
    matches = re.search(r"solution(?P<index>\d+)\.json", name)
    if matches is None:
        return False
    index = matches.group('index')
    if index is None:
        return False
    return True

def getInstanceAnalyses(instance, folder, solutionSampleSize = None, showResults = False):
    solutions = []
    solverResults = None
    fileNames = list(filter(fileIsSolution, os.listdir(folder)))
    if (solutionSampleSize is not None and solutionSampleSize > 0):
        fileNames = list(np.random.choice(fileNames, size=min(solutionSampleSize, len(fileNames)), replace=False))

    with open(join(folder, 'finished.json'), 'r') as f:
        solverResults = json.load(f)
    for name in fileNames:
        with open(join(folder, name), 'r') as f:
            data = json.load(f)
            data['name'] = name
            data['tiles'] = [{
                'coord': t['coord'],
                'tile': {
                    "id": str(t['tile']['id']),
                    "subTiles": [str(t['tile']['id'])],
                    "subTileConfiguration": [str(t['tile']['id'])],
                    "rotated": t['tile']['rotated'],
                    "X": t['tile']['X'],
                    "Y": t['tile']['Y']
                }
            } for t in data['tiles']]
            solutions.append(data)

    # build indexes
    tiles = solverResults['puzzle']['tiles']
    tiles.sort(key=lambda x: x['id'])
    indexes = flatten([[f'X[{tile["id"]}]', f'Y[{tile["id"]}]'] for tile in tiles])
    perfectRecursiveSolutionId = ",".join(sorted(str(tile["id"]) for tile in tiles))

    squaredNessRatios = []
    sameSidedEdges = 0
    allSuperTiles = dict()
    adjacency = np.zeros(shape=(10000, 10000))

    # count the number of same sided tiles
    edges = flatten([
        [(tile['id'], tile['X']), (tile['id'], tile['Y'])]
        for tile
        in tiles
    ])
    for (id1, es1) in edges:
        for (id2, es2) in edges:
            if id1 != id2 and es1 == es2:
                sameSidedEdges += 1
    sameSidedEdges /= 2

    for tile in tiles:
        minSize = min(tile['X'], tile['Y'])
        maxSize = max(tile['X'], tile['Y'])
        squaredNessRatios.append(minSize / maxSize)

    # evaluate the solutions
    for solution in solutions:
        matchedEdgesForSolution = []
        foundSuperTilesForSolution = []
        def evaluateSolution(placedTiles):
            superTiles = []
            edges = flatten([getPlacedTileEdges(placedTile) for placedTile in placedTiles])
            edgeDict = {}
            for edge in edges:
                (axis, eid, rotated, id, c1, c2) = edge
                edgeAsString = f'{c1} : {c2}'
                if edgeAsString not in edgeDict:
                    edgeDict[edgeAsString] = []
                edgeDict[edgeAsString].append(edge)

            for (axis, eid, rotated, id, c1, c2) in edges:
                edgeAsString = f'{c1} : {c2}'
                matchedEdges = edgeDict.get(edgeAsString, [])
                for (oaxis, oeid, orotated, oid, oc1, oc2) in matchedEdges:
                    if (eid != oeid and c1 == oc1 and c2 == oc2):
                        # record edge collision
                        collision = sorted([eid, oeid])
                        if collision not in matchedEdgesForSolution:
                            matchedEdgesForSolution.append(collision)

                        sid1, sid2 = sorted([id, oid])
                        superTile = [axis, sid1, sid2]
                        if superTile not in superTiles:
                            superTiles.append(superTile)

            for (axis, id, oid) in superTiles:
                # create a new solution with this superBlock
                tile1 = next(t for t in placedTiles if t['tile']['id'] == id)
                tile2 = next(t for t in placedTiles if t['tile']['id'] == oid)
                totalSubTiles = sorted(tile1['tile']['subTiles'] + tile2['tile']['subTiles'])
                newTileId = ",".join(totalSubTiles)

                newCoord = {
                    'X': min(tile1['coord']['X'], tile2['coord']['X']),
                    'Y': min(tile1['coord']['Y'], tile2['coord']['Y']),
                }
                newTile = {
                    "id": newTileId,
                    "subTiles": totalSubTiles,
                    "subTileConfiguration": [tile1['tile']['subTileConfiguration'], tile2['tile']['subTileConfiguration']],
                    "rotated": False,
                    "X": tile1['tile']['X'] if axis == 'X' else tile1['tile']['X'] + tile2['tile']['X'],
                    "Y": tile1['tile']['Y'] if axis == 'Y' else tile1['tile']['Y'] + tile2['tile']['Y']
                }

                superTileId = newTileId
                if superTileId not in foundSuperTilesForSolution:
                    foundSuperTilesForSolution.append(superTileId)

                newPlacements = [
                    { 'coord': newCoord, 'tile': newTile },
                    *filter(lambda x: x['tile']['id'] not in [id, oid], placedTiles)
                ]
                evaluateSolution(newPlacements)


        evaluateSolution(solution['tiles'])
        for (eid, oeid) in matchedEdgesForSolution:
            if (eid not in indexes):
                indexes.append(eid)
            if (oeid not in indexes):
                indexes.append(oeid)

            i1 = indexes.index(eid)
            i2 = indexes.index(oeid)
            adjacency[i1][i2] += 1
            adjacency[i2][i1] += 1

        for superTileId in foundSuperTilesForSolution:
            allSuperTiles[superTileId] = allSuperTiles.get(superTileId, 0) + 1

    # create the graph
    size = len(indexes)
    mesh = adjacency[0:size, 0:size]
    perfectRecursiveSolutions = allSuperTiles.get(perfectRecursiveSolutionId, 0)

    if showResults:
        fig, axes = plt.subplot_mosaic("AAC;AAB;AAB", constrained_layout=True)
        plot1 = axes['A']
        plot2 = axes['B']
        plot3 = axes['C']


        plot1.pcolormesh(mesh, edgecolors='k', linewidth=2, cmap='Reds')
        plot1.set_xlim([0, mesh.shape[0]])
        plot1.set_ylim([0, mesh.shape[1]])
        plot1.set_xticks(np.arange(0.5, size, 1), indexes, rotation = 45)
        plot1.set_yticks(np.arange(0.5, size, 1), indexes)
        for x in np.arange(0, size):
            for y in np.arange(0, size):
                if adjacency[x][y] > 0:
                    plot1.text(
                        x + 0.5,
                        y + 0.5,
                        str(adjacency[x][y]),
                        horizontalalignment='center',
                        verticalalignment='center',
                    )

        plot2.axis('tight')
        plot2.axis('off')
        cols=("superblock", "occurrences")
        rows=[(k,v) for k,v in allSuperTiles.items()]
        table = plot2.table(cellText=rows, colLabels=cols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)


        plot3.axis('tight')
        plot3.axis('off')
        rows=[
            ('solutions', solverResults['solutions']),
            ('fw', solverResults['puzzle']['fw']),
            ('fh', solverResults['puzzle']['fh']),
            ('# tiles', len(solverResults['puzzle']['tiles'])),
            ('# =edges', sameSidedEdges),
            ('rect. ratio', np.mean(squaredNessRatios)),
            ('# perfect recursive solutions', perfectRecursiveSolutions)
        ]
        table = plot3.table(cellText=rows, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        plt.show()

    return {
        'name': instance,
        'numberOfTiles': len(solverResults['puzzle']['tiles']),
        'numberOfSameSidedEdges': sameSidedEdges,
        'squaredNessRatio': squaredNessRatios,
        'numberOfSolutions': solverResults['solutions'],
        'numberOfEvaluatedSolutions': len(solutions),
        'numberOfPerfectRecursiveSolutions': perfectRecursiveSolutions,
        'superBlocks': allSuperTiles,
        'adjacencyMesh': mesh.tolist(),
        'indexes': indexes,
        'csvLine': solverResults['csvLine']
    }

def getInstanceAnalysesStar(args):
    return getInstanceAnalyses(*args)

def main():
#     getInstanceAnalyses('2', '../results/06/2', None, True)
#     return

    print("generating adjacency map")
    groupName = str(sys.argv[1])
    shouldGenerate = sys.argv[2] == "true"
    solutionSampleSize = int(sys.argv[3])

    resultsFolder = "../results/" + groupName
    csvName = "./data/" + groupName + "tiles.csv"

    if (shouldGenerate):
        results = []
        instances = [x for x in os.listdir(resultsFolder) if x.isnumeric()]
        instanceFolders = [join(resultsFolder, instance) for instance in instances]
        inputs = list(zip(instances, instanceFolders, [solutionSampleSize for x in instances]))
        with Pool(6) as pool:
            results = list(tqdm(pool.imap_unordered(getInstanceAnalysesStar, inputs, chunksize=1), total=len(inputs)))
        with open(join(resultsFolder, 'analyses.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # visualize the results
    results = None
    with open(join(resultsFolder, 'analyses.json'), 'r') as f:
        results = json.load(f)

#     print("puzzle without superblocks:")
#     print([(x['name'], x['numberOfSolutions']) for x in results if len(x['superBlocks']) == 0])


    # show my results
    fig, axes = plt.subplot_mosaic("ABCE;DDDD", constrained_layout=True)

    plt1 = axes['A']
    plt2 = axes['B']
    plt3 = axes['C']
    plt4 = axes['D']
    plt5 = axes['E']

    plt1.scatter([x['numberOfSameSidedEdges'] for x in results], [x['numberOfSolutions'] for x in results])
    plt1.set_xlabel('number of same sided edges')
    plt1.set_ylabel('number of solutions')

    plt5.scatter([len(x['superBlocks']) for x in results], [x['numberOfSolutions'] for x in results])
    plt5.set_xlabel('number of superblocks')
    plt5.set_ylabel('number of solutions')

    plt2.scatter([x['numberOfPerfectRecursiveSolutions'] for x in results], [x['numberOfSolutions'] for x in results])
    plt2.set_xlabel('number of perfect recursive solutions')
    plt2.set_ylabel('number of solutions')

    plt3.errorbar(
        [np.mean(x['squaredNessRatio']) for x in results],
        [x['numberOfSolutions'] for x in results],
        xerr=[np.std(x['squaredNessRatio']) for x in results],
        fmt='o',
        ecolor = 'r'
    )
    plt3.set_xlabel('mean squaredness ratio (short side / long side)')
    plt3.set_ylabel('number of solutions')


    Values = [x for x in results if x['numberOfSolutions'] > 0]
    X_axis = np.arange(len(Values))

    plt4.bar(X_axis - 0.2, [x['numberOfPerfectRecursiveSolutions'] for x in Values], 0.4, label = 'Perfect solutions')
    plt4.bar(X_axis + 0.2, [x['numberOfEvaluatedSolutions'] for x in Values], 0.4, label = 'Solutions')
    plt4.set_xlim(-0.5, len(Values) - 0.5)
    plt4.set_xlabel('Instance')
    plt4.set_ylabel('number of solutions')
    plt4.set_xticks(X_axis, [x['name'] for x in Values], rotation = 45)
    plt4.legend()

    plt.show()

    # relate to the dataset
    fig, axes = plt.subplot_mosaic("AB;CD", constrained_layout=True)

    plt1 = axes['A']
    plt2 = axes['B']
    plt3 = axes['C']
    plt4 = axes['D']

    plt1.scatter([x['numberOfSolutions'] for x in results], [int(x['csvLine']['recursions']) for x in results])
    plt1.set_xlabel('number of solutions')
    plt1.set_ylabel('number of recursions')

    plt2.scatter([len(x['superBlocks']) for x in results], [int(x['csvLine']['recursions']) for x in results])
    plt2.set_xlabel('number of superblocks')
    plt2.set_ylabel('number of recursions')

    plt3.scatter([x['numberOfPerfectRecursiveSolutions'] for x in results], [int(x['csvLine']['recursions']) for x in results])
    plt3.set_xlabel('number of perfect recursive solutions')
    plt3.set_ylabel('number of recursions')

    plt4.scatter([x['numberOfSameSidedEdges'] for x in results], [int(x['csvLine']['recursions']) for x in results])
    plt4.set_xlabel('number of same sided edges')
    plt4.set_ylabel('number of recursions')

    plt.show()


if __name__ == '__main__':
    main()