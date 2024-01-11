To run the solver use
> mvn package
> java -jar solver.jar <group: 06/07/08/09/...>

To run the visualizer use
> python3 ./src/main/kotlin/org/example/generatePlots.py <path to instance: ../results/06/4>

To run the instance analyser + problem analyser
> python3 ./src/main/kotlin/org/example/generateAdjacencyGraphs.py <group: 06/07/08/09/...> true 0

To run only the problem analyser 
> python3 ./src/main/kotlin/org/example/generateAdjacencyGraphs.py <group: 06/07/08/09/...> false 0

To the problem analyser for multiple groups
> for i in 06 07 08 09 10; do python3 ./src/main/kotlin/org/example/generateAdjacencyGraphs.py "$i" truel 0; done
