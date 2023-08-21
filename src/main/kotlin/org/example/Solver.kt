package org.example

import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import kotlinx.coroutines.*
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlin.system.measureTimeMillis

data class RuntimeState(val mapper: ObjectMapper, var solutions: AtomicInteger)
data class Tile(
    @field:JsonProperty("id") val id: Int,
    @field:JsonProperty("rotated") val rotated: Boolean,
    @field:JsonProperty("X") val x: Int,
    @field:JsonProperty("Y") val y: Int
)
data class Coord(@field:JsonProperty("X") val x: Int, @field:JsonProperty("Y") val y: Int)

typealias Matrix2D<T> = Array<Array<T>>

inline fun <reified T> createMatrix(x: Int, y: Int, v: T): Matrix2D<T> {
    return Array(x) { Array(y) { v } }
}

fun getIndicesForMatrix(width: Int, height: Int) = sequence<Coord> {
    for (y in 0 until height) {
        for (x in 0 until width) {
            yield(Coord(x, y))
        }
    }
}

fun add_at(m1: Matrix2D<Int>, pos: Coord, tile: Tile, v: Int) {
    assert(pos.x + tile.x < m1.size) { "the position will make the small matrix exceed the boundaries at x" }
    assert(pos.y + tile.y < m1[0].size) { "the position will make the small matrix exceed the boundaries at y" }
    for (x in pos.x until pos.x + tile.x) {
        for (y in pos.y until pos.y + tile.y) {
            m1[x][y] += v
        }
    }
}

fun saveSolution(groupName: String, state: RuntimeState, id: Int, fw: Int, fh: Int, usedTiles: List<Pair<Coord, Tile>>) {
    val board = createMatrix(fw, fh, 0)
    for (placement in usedTiles) {
        val (pos, tile) = placement
        add_at(board, pos, tile, tile.id)
    }

    val solutionIndex = state.solutions.getAndIncrement()
    File("../results/$groupName/${id}/").mkdirs()
    File("../results/$groupName/${id}/solution${solutionIndex}.json").writeText(state.mapper.writerWithDefaultPrettyPrinter().writeValueAsString(
        mapOf(
            "tiles" to usedTiles.map { (coord, tile) ->
                mapOf(
                    "coord" to coord,
                    "tile" to tile
                )
            },
            "board" to board
        )
    ))
}


fun markFinished(groupName: String, state: RuntimeState, id: Int, runtime: Long, fw: Int, fh: Int, tiles: List<Tile>, puzzle: Map<String, String>) {
    File("../results/$groupName/${id}/").mkdirs()
    File("../results/$groupName/${id}/finished.json").writeText(state.mapper.writerWithDefaultPrettyPrinter().writeValueAsString(
        mapOf(
            "solutions" to state.solutions.get(),
            "runtime" to runtime,
            "csvLine" to puzzle,
            "puzzle" to mapOf(
                "id" to puzzle["id"]!!,
                "fw" to fw,
                "fh" to fh,
                "tiles" to tiles
            )
        )
    ))
//    Runtime.getRuntime().exec("python3 ./src/main/kotlin/org/example/generatePlots.py ./results/${id}").waitFor()
}

suspend fun solvePuzzle(groupName: String, state: RuntimeState, id: Int, fw: Int, fh: Int, remainingTiles: List<Tile>, usedTiles: List<Pair<Coord, Tile>>) {
    withContext(Dispatchers.Default) {
        // sum the tiles to find overlap
        val board = createMatrix(fw, fh, 0)
        for ((pos, tile) in usedTiles) {
            add_at(board, pos, tile, 1)
        }
        if (board.any { it.any { it > 1 } }) {
            // overlap, ignore
            return@withContext
        }

        if (remainingTiles.isEmpty()) {
            // if no overlap, and no more tiles then we found a solution
            saveSolution(groupName, state, id, fw, fh, usedTiles)
            return@withContext
        }

        // attempt to sub solve
        val newRemainingTiles = remainingTiles.toMutableList()

        // first try one orientation
        val tileNormal = newRemainingTiles.removeFirst()
        val normal = getIndicesForMatrix(fw - tileNormal.x + 1, fh - tileNormal.y + 1).mapNotNull { pos ->
            if (board[pos.x][pos.y] > 0) {
                // prevent overlap
                return@mapNotNull null
            }
            add_at(board, pos, tileNormal, 1)
            val causedOverlap = board.any { it.any { it > 1 } }
            add_at(board, pos, tileNormal, -1)
            if (causedOverlap) {
                // prevent overlap
                return@mapNotNull null
            }

            // actually iterate deeper
            async {
                solvePuzzle(groupName, state, id, fw, fh, newRemainingTiles, usedTiles + listOf(Pair(pos, tileNormal)))
            }
        }

        //secondly try the rotated orientation
        val tileRotated = Tile(tileNormal.id, true, tileNormal.y, tileNormal.x)
        val rotated = getIndicesForMatrix(fw - tileRotated.x + 1, fh - tileRotated.y + 1).mapNotNull { pos ->
            if (board[pos.x][pos.y] > 0) {
                // prevent overlap
                return@mapNotNull null
            }
            add_at(board, pos, tileRotated, 1)
            val causedOverlap = board.any { it.any { it > 1 } }
            add_at(board, pos, tileRotated, -1)
            if (causedOverlap) {
                // prevent overlap
                return@mapNotNull null
            }

            // actually iterate deeper
            async {
                solvePuzzle(groupName, state, id, fw, fh, newRemainingTiles, usedTiles + listOf(Pair(pos, tileRotated)))
            }
        }

        // wait for all entries to finish
        (normal.toList() + rotated.toList()).awaitAll()
    }
}

fun processPuzzle(groupName: String, puzzle: Map<String, String>) {
    val mapper = jacksonObjectMapper()
    val id = puzzle["id"]!!.toInt()
    val fw = puzzle["frame_width"]!!.toInt()
    val fh = puzzle["frame_height"]!!.toInt()
    val hasSolution = puzzle["has_solutions"] == "1"
    val tiles = mapper.readValue(
        puzzle["tiles"]!!,
        object : TypeReference<List<Map<String, String>>>(){}
    ).mapIndexed { index, it ->
        Tile(
            id = index + 1,
            rotated = false,
            x = it["X"]!!.toInt(),
            y = it["Y"]!!.toInt()
        )
    }
    val sortedTiles = tiles.sortedBy { tile -> tile.x * tile.y }.reversed()

    if (File("./results/$id/finished.json").exists()) {
        println("already solved puzzle $id")
        return
    }

    if (!hasSolution) {
        println("puzzle $id has no solution")
        val state = RuntimeState(mapper, AtomicInteger(0))
        markFinished(groupName, state, id, 0, fw, fh, tiles, puzzle)
    } else {
        println("solving puzzle $id")
        val state = RuntimeState(mapper, AtomicInteger(0))
        val millis = measureTimeMillis {
            runBlocking(Dispatchers.Default) {
                solvePuzzle(groupName, state, id, fw, fh, sortedTiles, emptyList())
            }
        }
        println("took $millis ms to solve puzzle ${puzzle["id"]}")
        markFinished(groupName, state, id, millis, fw, fh, tiles, puzzle)
    }
}

fun main(args: Array<String>) {
    println(Runtime.getRuntime().maxMemory());

    // solve
    val groupName = "09"
    val rows = csvReader().readAllWithHeader(File("./data/${groupName}tiles.csv"))
    println("will attempt to solve ${rows.size} puzzles")
    for (puzzle in rows) {
        processPuzzle(groupName, puzzle)
    }
    println("finished")
}

