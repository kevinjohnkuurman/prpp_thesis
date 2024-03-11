# python3 ./src/main/kotlin/org/example/generateAdjacencyGraphs.py 06 true
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
import re
import itertools
from multiprocessing import Pool
from operator import itemgetter
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from os.path import isfile, join
from matplotlib.widgets import Slider


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def isPerfectSuperTile(conf):
    if len(conf) != 2:
        return False

    (p1, p2) = sorted(conf, key=lambda x: len(x))
    return len(p1) == 1 and (len(p2) == 1 or (len(p2) == 2 and isPerfectSuperTile(p2)))


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

    c1 = (coord['X'], coord['Y'])
    c2 = (coord['X'], coord['Y'] + tile['Y'])
    c3 = (coord['X'] + tile['X'], coord['Y'])
    c4 = (coord['X'] + tile['X'], coord['Y'] + tile['Y'])

    return [
        ('X', tx, rotated, id, c1, c3),
        ('X', tx, rotated, id, c2, c4),
        ('Y', ty, rotated, id, c1, c2),
        ('Y', ty, rotated, id, c3, c4),
    ]


def getInstanceAnalyses(instance, folder, solutionSampleSize=None, showResults=False):
    solutions = []
    solverResults = None
    with open(join(folder, 'finished.json'), 'r') as f:
        solverResults = json.load(f)
    for idx, data in enumerate(solverResults['foundSolutions']):
        data['name'] = str(idx)
        data['tiles'] = [{
            'coord': t['coord'],
            'tile': {
                "id": str(t['tile']['id']),
                "configId": subPathToString([str(t['tile']['id'])]),
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
    pureIndexes = indexes[:]
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
    totalPerfectRecursiveSolutions = 0
    for solution in solutions:
        foundConfigurations = {}
        matchedEdgesForSolution = []
        foundSuperTilesForSolution = []
        foundPerfectSolutionSuperTiles = []

        def evaluateSolution(placedTiles):
            superTilesForStep = []
            edgesForStep = flatten([getPlacedTileEdges(placedTile) for placedTile in placedTiles])
            edgeDict = {}
            for edge in edgesForStep:
                (axis, eid, rotated, id, c1, c2) = edge
                edgeAsString = f'{c1} : {c2}'
                if edgeAsString not in edgeDict:
                    edgeDict[edgeAsString] = []
                edgeDict[edgeAsString].append(edge)

            for (axis, eid, rotated, id, c1, c2) in edgesForStep:
                edgeAsString = f'{c1} : {c2}'
                matchedEdges = edgeDict.get(edgeAsString, [])
                for (oaxis, oeid, orotated, oid, oc1, oc2) in matchedEdges:
                    if eid != oeid and c1 == oc1 and c2 == oc2:
                        # record edge collision
                        collision = sorted([eid, oeid])
                        if collision not in matchedEdgesForSolution:
                            matchedEdgesForSolution.append(collision)

                        sid1, sid2 = sorted([id, oid])
                        superTile = [axis, sid1, sid2]
                        if superTile not in superTilesForStep:
                            superTilesForStep.append(superTile)

            for (axis, id, oid) in superTilesForStep:
                # create a new solution with this supertiles
                tile1 = next(t for t in placedTiles if t['tile']['id'] == id)
                tile2 = next(t for t in placedTiles if t['tile']['id'] == oid)
                totalSubTiles = sorted(tile1['tile']['subTiles'] + tile2['tile']['subTiles'])
                newConfiguration = [tile1['tile']['subTileConfiguration'], tile2['tile']['subTileConfiguration']]
                newTileId = ",".join(totalSubTiles)
                newConfigurationId = subPathToString(newConfiguration)

                newCoord = {
                    'X': min(tile1['coord']['X'], tile2['coord']['X']),
                    'Y': min(tile1['coord']['Y'], tile2['coord']['Y']),
                }
                newTile = {
                    "id": newTileId,
                    "configId": newConfigurationId,
                    "subTiles": totalSubTiles,
                    "subTileConfiguration": newConfiguration,
                    "rotated": False,
                    "X": tile1['tile']['X'] if axis == 'X' else tile1['tile']['X'] + tile2['tile']['X'],
                    "Y": tile1['tile']['Y'] if axis == 'Y' else tile1['tile']['Y'] + tile2['tile']['Y']
                }

                newPlacements = [
                    {'coord': newCoord, 'tile': newTile},
                    *filter(lambda x: x['tile']['id'] not in [id, oid], placedTiles)
                ]

                # record the new super-tile appearance
                if newTileId not in foundSuperTilesForSolution:
                    foundSuperTilesForSolution.append(newTileId)

                # record this configuration as a perfect solution
                if len(newPlacements) == 1 and isPerfectSuperTile(newTile["subTileConfiguration"]) and newTileId not in foundPerfectSolutionSuperTiles:
                    # print(solution["name"])
                    foundPerfectSolutionSuperTiles.append(newTileId)

                # if the same configuration appears multiple times we can skip it, we will not find more the second time around
                configurationId = ":::".join(sorted([tile['tile']['configId'] for tile in newPlacements]))
                if len(newPlacements) > 1 and configurationId not in foundConfigurations:
                    foundConfigurations[configurationId] = True
                    evaluateSolution(newPlacements)

        evaluateSolution(solution['tiles'])
        for (eid, oeid) in matchedEdgesForSolution:
            if eid not in indexes:
                indexes.append(eid)
            if oeid not in indexes:
                indexes.append(oeid)

            i1 = indexes.index(eid)
            i2 = indexes.index(oeid)
            adjacency[i1][i2] += 1
            adjacency[i2][i1] += 1

        for superTileId in foundSuperTilesForSolution:
            allSuperTiles[superTileId] = allSuperTiles.get(superTileId, 0) + 1

        totalPerfectRecursiveSolutions += len(foundPerfectSolutionSuperTiles)

    # create the graph
    size = len(indexes)
    mesh = adjacency[0:size, 0:size]
    numberOfRecursiveSolutions = allSuperTiles.get(perfectRecursiveSolutionId, 0)
    numberOfPerfectRecursiveSolutions = totalPerfectRecursiveSolutions

    return {
        'name': instance,
        'numberOfTiles': len(solverResults['puzzle']['tiles']),
        'numberOfSameSidedEdges': sameSidedEdges,
        'squaredNessRatio': squaredNessRatios,
        'numberOfSolutions': solverResults['solutions'],
        'numberOfEvaluatedSolutions': len(solutions),
        'numberOfRecursiveSolutions': numberOfRecursiveSolutions,
        'numberOfPerfectRecursiveSolutions': numberOfPerfectRecursiveSolutions,
        'superTiles': allSuperTiles,
        'adjacencyMesh': mesh.tolist(),
        'pureIndexes': pureIndexes,
        'indexes': indexes,
        'csvLine': solverResults['csvLine']
    }


def getInstanceAnalysesStar(args):
    return getInstanceAnalyses(*args)


def main():
    groupName = str(sys.argv[1])
    shouldGenerate = sys.argv[2] == "true"
    solutionSampleSize = int(sys.argv[3])
    print("generating adjacency map " + groupName)

    resultsFolder = "../results/" + groupName
    csvName = "./data/" + groupName + "tiles.csv"

    # getInstanceAnalyses('4602', join(resultsFolder, '4602'), 0, None)
    # return

    if shouldGenerate:
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

    # generate the graphs
    plt.rcParams.update({'font.size': 14})
    C_SOLVABLE = np.array([1, 84, 166]) / 255.0
    C_UNSOLVABLE = np.array([238, 28, 37]) / 255.0

    print((len(results), len([1 for x in results if x['numberOfSolutions'] > 0])))
    values = sorted([x for x in results if x['numberOfSolutions'] > 0], key=lambda x: x['numberOfEvaluatedSolutions'])
    values = [x for x in values if x['numberOfEvaluatedSolutions'] > 0]
    print(("least solutions", values[0]['name'], values[0]['numberOfSolutions']))
    print(("most solutions", values[-1]['name'], values[-1]['numberOfSolutions']))

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.scatter(
        [x['numberOfSolutions'] for x in results],
        [int(x['csvLine']['recursions']) for x in results],
        alpha=0.1,
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results]
    )
    plt1.set_xlabel('number of solutions')
    plt1.set_ylabel('instance hardness')
    plt1.tick_params(axis='x', labelrotation=45)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_solutions_hardness.jpg")

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.errorbar(
        [np.mean(x['squaredNessRatio']) for x in results if int(x['csvLine']['has_solutions']) == 1],
        [x['numberOfSolutions'] for x in results if int(x['csvLine']['has_solutions']) == 1],
        color=C_SOLVABLE,
        ecolor=C_SOLVABLE,
        xerr=[np.std(x['squaredNessRatio']) for x in results if int(x['csvLine']['has_solutions']) == 1],
        fmt='o',
        alpha=0.1,
    )
    plt1.errorbar(
        [np.mean(x['squaredNessRatio']) for x in results if int(x['csvLine']['has_solutions']) == 0],
        [x['numberOfSolutions'] for x in results if int(x['csvLine']['has_solutions']) == 0],
        color=C_UNSOLVABLE,
        ecolor=C_UNSOLVABLE,
        xerr=[np.std(x['squaredNessRatio']) for x in results if int(x['csvLine']['has_solutions']) == 0],
        fmt='o',
        alpha=0.1,
    )
    plt1.set_xlabel('mean squareness ratio (short side / long side)')
    plt1.set_ylabel('number of solutions')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=2, handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_squareness.jpg")

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt1.scatter(
        [x['numberOfSameSidedEdges'] for x in results],
        [x['numberOfSolutions'] for x in results],
        alpha=0.1,
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results],
    )
    plt1.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    plt1.set_ylabel('number of solutions')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_same_sided_edges.jpg")

    # fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    # plt1 = axes['A']
    # values = sorted([x for x in results if x['numberOfSolutions'] > 0], key=lambda x: x['numberOfEvaluatedSolutions'])
    # values = [x for x in values if x['numberOfEvaluatedSolutions'] > 0]
    # print(("least solutions", values[0]['name']))
    # print(("most solutions", values[-1]['name']))
    # N = min([50, len(values)])
    # values = [values[i] for i in np.linspace(0.5, len(values) - 0.5, N, dtype=int)]
    # X_axis = np.arange(len(values))
    #
    # plt1.bar(X_axis - 0.2, [x['numberOfRecursiveSolutions'] for x in values], 0.4, label='Recursive super-tile solutions')
    # plt1.bar(X_axis + 0.2, [x['numberOfEvaluatedSolutions'] for x in values], 0.4, label='Solutions')
    # plt1.set_xlim(0, len(values))
    # plt1.set_xlabel('Instance')
    # plt1.set_xticks(X_axis, ['' for x in X_axis])
    # plt1.legend(loc='upper left')
    # plt.savefig("./figures/" + groupName + "_solutions_vs_recursive_solutions.jpg", dpi=200)

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.scatter(
        [x['numberOfSolutions'] for x in results],
        [x['numberOfRecursiveSolutions'] for x in results],
        alpha=0.1,
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results]
    )
    plt1.set_xlabel('number of solutions')
    plt1.set_ylabel('number of recursive super-tile solutions')
    plt1.tick_params(axis='x', labelrotation=45)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_solutions_vs_recursive_solutions.jpg", dpi=200)

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt1.scatter(
        [x['numberOfSameSidedEdges'] for x in results],
        [int(x['csvLine']['recursions']) for x in results],
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results],
        alpha=0.1
    )
    plt1.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    plt1.set_ylabel('hardness')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=2, handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_hardness_same_sided_edges.jpg")

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.scatter(
        [np.mean(x['squaredNessRatio']) for x in results],
        [int(x['csvLine']['recursions']) for x in results],
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results],
        alpha=0.1
    )
    plt1.set_xlabel('mean squareness ratio')
    plt1.set_ylabel('hardness')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_squareness_hardness.jpg")

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt1.scatter(
        [len(x['superTiles']) for x in results],
        [int(x['csvLine']['recursions']) for x in results],
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results],
        alpha=0.1
    )
    plt1.set_xlabel('number of composable super-tiles')
    plt1.set_ylabel('hardness')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_all_supertiles_hardness.jpg")

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt1.scatter(
        [len([y for y in x['superTiles'] if len(y.split(",")) == 2]) for x in results],
        [int(x['csvLine']['recursions']) for x in results],
        c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results],
        alpha=0.1
    )
    plt1.set_xlabel('number of composable first degree super-tiles')
    plt1.set_ylabel('hardness')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='With solutions', markerfacecolor=C_SOLVABLE, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Without solutions', markerfacecolor=C_UNSOLVABLE, markersize=10),
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./figures/" + groupName + "_first_degree_supertiles_hardness.jpg")

    # fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    # plt1 = axes['A']
    # plt1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt1.scatter(
    #     [x['numberOfSameSidedEdges'] for x in results],
    #     [int(x['csvLine']['recursions']) for x in results],
    #     c=[C_SOLVABLE if int(x['csvLine']['has_solutions']) == 1 else C_UNSOLVABLE for x in results],
    #     alpha=0.1
    # )
    # plt1.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    # plt1.set_ylabel('hardness')
    # plt.savefig("./figures/" + groupName + "_same_sided_edges_hardness.jpg")

    if False:
        instance = next(x for x in results if x['name'] == '4661')
        print(instance)
        indexes = instance['pureIndexes']
        adjacency = np.array(instance['adjacencyMesh'])
        size = len(indexes)
        def conv(v):
            (edge, id) = v.rstrip(v[-1]).split("[")
            return f"${id} ${edge}"
        convertedIndexes = [conv(x) for x in indexes]
        mesh = adjacency[0:size, 0:size]

        fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
        plot1 = axes['A']

        viridis = cm.get_cmap('Greys', 256)
        newcolors = viridis(np.linspace(0, 1, 256))[0:100]
        newcmp = ListedColormap(newcolors)

        plot1.pcolormesh(mesh, edgecolors='k', linewidth=2, cmap=newcmp)
        plot1.set_xlim([0, mesh.shape[0]])
        plot1.set_ylim([0, mesh.shape[1]])
        plot1.set_xticks(np.arange(0.5, size, 1), convertedIndexes, rotation=45)
        plot1.set_yticks(np.arange(0.5, size, 1), convertedIndexes)
        plot1.set_xlabel('Tile 1')
        plot1.set_ylabel('Tile 2')
        for x in np.arange(0, size):
            for y in np.arange(0, size):
                if adjacency[x][y] > 0:
                    plot1.text(
                        x + 0.5,
                        y + 0.5,
                        str(int(adjacency[x][y])),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize='12'
                        )
        plt.gcf().set_size_inches(10, 10)
        plt.savefig("./figures/" + groupName + "_instance_analyses.jpg", dpi = 200)


    if False:
        dots = []
        for x in ["06", "07", "08", "09", "10"]:
            with open(join("../results/" + x, 'analyses.json'), 'r') as f:
                results = json.load(f)
            dots = dots + [(x['numberOfTiles'] - 5, x['numberOfSolutions']) for x in results]
        fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
        plt1 = axes['A']
        plt1.scatter(
            [x[0] for x in dots],
            [x[1] for x in dots],
            alpha=0.1
        )
        plt1.set_xlabel('number of tiles')
        plt1.set_ylabel('number of solutions')
        plt1.set_xticks([1, 2, 3, 4, 5], ["6", "7", "8", "9", "10"])
        plt.savefig("./figures/growth.jpg")

    exit(0)

    # show my results
    fig, axes = plt.subplot_mosaic("AB;CE", constrained_layout=True)

    plt1 = axes['A']
    plt2 = axes['B']
    plt3 = axes['C']
    plt5 = axes['E']

    plt1.scatter([x['numberOfSameSidedEdges'] for x in results], [x['numberOfSolutions'] for x in results], alpha=0.1)
    plt1.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    plt1.set_ylabel('number of solutions')

    plt5.scatter([len(x['superTiles']) for x in results], [x['numberOfSolutions'] for x in results], alpha=0.1)
    plt5.set_xlabel('number of super-tiles')
    plt5.set_ylabel('number of solutions')

    plt2.scatter([x['numberOfRecursiveSolutions'] for x in results], [x['numberOfSolutions'] for x in results],
                 alpha=0.1)
    plt2.set_xlabel('number of recursive solutions')
    plt2.set_ylabel('number of solutions')

    plt3.errorbar(
        [np.mean(x['squaredNessRatio']) for x in results],
        [x['numberOfSolutions'] for x in results],
        xerr=[np.std(x['squaredNessRatio']) for x in results],
        fmt='o',
        ecolor='r'
    )
    plt3.set_xlabel('mean squaredness ratio (short side / long side)')
    plt3.set_ylabel('number of solutions')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([x['numberOfRecursiveSolutions'] for x in results],
               [x['numberOfSolutions'] for x in results],
               [x['numberOfPerfectRecursiveSolutions'] for x in results],
               alpha=0.1)
    ax.set_xlabel('number of recursive solutions')
    ax.set_ylabel('number of solutions')
    ax.set_zlabel('number of perfect recursive solutions')
    plt.show()

    fig, axes = plt.subplot_mosaic("A", constrained_layout=True)
    plt1 = axes['A']
    Values = sorted([x for x in results if x['numberOfSolutions'] > 0], key=lambda x: x['numberOfEvaluatedSolutions'])
    X_axis = np.arange(len(Values))

    plt1.bar(X_axis - 0.2, [x['numberOfPerfectRecursiveSolutions'] for x in Values], 0.2, label='Perfect recursive solutions')
    plt1.bar(X_axis,       [x['numberOfRecursiveSolutions'] for x in Values], 0.2, label='Recursive solutions')
    plt1.bar(X_axis + 0.2, [x['numberOfEvaluatedSolutions'] for x in Values], 0.2, label='Solutions')
    plt1.set_xlim(0, len(Values))
    plt1.set_xlabel('Instance')
    plt1.set_ylabel('number of solutions')
    plt1.set_xticks(X_axis, [x['name'] for x in Values], rotation=90)
    plt1.legend()
    plt.show()

    # relate to the dataset
    fig, axes = plt.subplot_mosaic("AB;CD", constrained_layout=True)

    plt1 = axes['A']
    plt2 = axes['B']
    plt3 = axes['C']
    plt4 = axes['D']

    plt1.scatter([x['numberOfSolutions'] for x in results], [int(x['csvLine']['recursions']) for x in results],
                 alpha=0.1)
    plt1.set_xlabel('number of solutions')
    plt1.set_ylabel('number of recursions')

    plt2.scatter([len(x['superTiles']) for x in results], [int(x['csvLine']['recursions']) for x in results],
                 alpha=0.1)
    plt2.set_xlabel('number of superTiles')
    plt2.set_ylabel('number of recursions')

    plt3.scatter([x['numberOfRecursiveSolutions'] for x in results],
                 [int(x['csvLine']['recursions']) for x in results], alpha=0.1)
    plt3.set_xlabel('number of recursive solutions')
    plt3.set_ylabel('number of recursions')

    plt4.scatter([x['numberOfSameSidedEdges'] for x in results], [int(x['csvLine']['recursions']) for x in results],
                 alpha=0.1)
    plt4.set_xlabel('number of tile pairs sharing an edge of equal dimension')
    plt4.set_ylabel('number of recursions')

    if True:
        instance = next(x for x in results if x['name'] == '4661')
        print(instance)
        indexes = instance['pureIndexes']
        adjacency = np.array(instance['adjacencyMesh'])
        allSuperTiles = instance['superTiles']
        numberOfSolutions = instance['numberOfSolutions']
        numberOfRecursiveSolutions = instance['numberOfRecursiveSolutions']
        numberOfPerfectRecursiveSolutions = instance['numberOfPerfectRecursiveSolutions']
        size = len(indexes)
        mesh = adjacency[0:size, 0:size]

        fig, axes = plt.subplot_mosaic("AAC;AAB;AAB", constrained_layout=True)
        plot1 = axes['A']
        plot2 = axes['B']
        plot3 = axes['C']

        viridis = cm.get_cmap('GnBu', 256)
        newcolors = viridis(np.linspace(0, 1, 256))[0:200]
        newcmp = ListedColormap(newcolors)

        plot1.pcolormesh(mesh, edgecolors='k', linewidth=2, cmap=newcmp)
        plot1.set_xlim([0, mesh.shape[0]])
        plot1.set_ylim([0, mesh.shape[1]])
        plot1.set_xticks(np.arange(0.5, size, 1), indexes, rotation=45)
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

        print(allSuperTiles)
        plot2.axis('tight')
        plot2.axis('off')
        cols = ("super-tiles", "occurrences")
        rows = [(k, v) for k, v in allSuperTiles.items()]
        table = plot2.table(cellText=rows, colLabels=cols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        plot3.axis('tight')
        plot3.axis('off')
        rows = [
            ('solutions', numberOfSolutions),
            ('fw', instance['csvLine']['frame_width']),
            ('fh', instance['csvLine']['frame_height']),
            ('# tiles', len(instance['csvLine']['tiles'])),
            ('# =edges', instance['numberOfSameSidedEdges']),
            ('rect. ratio', np.mean(instance['squaredNessRatio'])),
            ('# recursive solutions', numberOfRecursiveSolutions),
            ('# perfect recursive solutions', numberOfPerfectRecursiveSolutions)
        ]
        table = plot3.table(cellText=rows, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        plt.show()




    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([len(x['superTiles']) for x in results],
               [x['numberOfSolutions'] for x in results],
               [int(x['csvLine']['recursions']) for x in results],
               alpha=0.1)
    ax.set_xlabel('number of super-tiles')
    ax.set_ylabel('number of solutions')
    ax.set_zlabel('number of recursions')
    plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter([x['numberOfSolutions'] for x in results], [int(x['csvLine']['recursions']) for x in results],
    #              alpha=0.1)
    # ax.set_xlabel('number of solutions')
    # ax.set_ylabel('number of recursions')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
