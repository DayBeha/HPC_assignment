"""
conway.py

A simple Python/matplotlib implementation of Conway's Game of Life.

Author: Mahesh Venkitachalam
Github: https://github.com/electronut/pp/blob/master/conway/conway.py
"""

import sys, argparse
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ON = 255
OFF = 0
vals = [ON, OFF]


def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[0.2, 0.8]).reshape(N, N)


# @profile
def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0, 0, 255], [255, 0, 255], [0, 255, 255]])
    grid[i: i + 3, j: j + 3] = glider


# @profile
def addGosperGliderGun(i, j, grid):
    """adds a Gosper Glider Gun with top left cell at (i, j)"""
    gun = np.zeros(11 * 38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 255
    gun[6][1] = gun[6][2] = 255

    gun[3][13] = gun[3][14] = 255
    gun[4][12] = gun[4][16] = 255
    gun[5][11] = gun[5][17] = 255
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 255
    gun[7][11] = gun[7][17] = 255
    gun[8][12] = gun[8][16] = 255
    gun[9][13] = gun[9][14] = 255

    gun[1][25] = 255
    gun[2][23] = gun[2][25] = 255
    gun[3][21] = gun[3][22] = 255
    gun[4][21] = gun[4][22] = 255
    gun[5][21] = gun[5][22] = 255
    gun[6][23] = gun[6][25] = 255
    gun[7][25] = 255

    gun[3][35] = gun[3][36] = 255
    gun[4][35] = gun[4][36] = 255

    grid[i: i + 11, j: j + 38] = gun


def update(frameNum, img, grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):
            # compute 8-neghbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulaton takes place on a toroidal surface.
            total = int(
                (
                        grid[i, (j - 1) % N]
                        + grid[i, (j + 1) % N]
                        + grid[(i - 1) % N, j]
                        + grid[(i + 1) % N, j]
                        + grid[(i - 1) % N, (j - 1) % N]
                        + grid[(i - 1) % N, (j + 1) % N]
                        + grid[(i + 1) % N, (j - 1) % N]
                        + grid[(i + 1) % N, (j + 1) % N]
                )
                / 255
            )
            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON
    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return (img,)


@profile
def update_(grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line
    newGrid = grid.copy()
    totals = np.zeros_like(grid)

    for i in range(N):
        for j in range(N):
            if i in [0, N - 1] and j in [0, N - 1]:
                totals[i, j] = int((grid[i, (j - 1) % N] + grid[i, (j + 1) % N] +
                                    grid[(i - 1) % N, j] + grid[(i + 1) % N, j] +
                                    grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1) % N] +
                                    grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1) % N]) / 255)
    totals[1:-1, 1:-1] += ((grid[:-2, :-2] + grid[:-2, 1:-1] + grid[:-2, 2:] +
                          grid[1:-1, :-2]              + grid[1:-1, 2:] +
                          grid[2:, :-2] + grid[2:, 1:-1] + grid[2:, 2:])/255).astype(np.int32)

    newGrid[np.logical_and(grid == ON, np.logical_or((totals < 2), (totals > 3)))] = OFF
    newGrid[np.logical_and(grid != ON, totals == 3)] = ON

    return newGrid


# main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Runs Conway's Game of Life simulation."
    )
    # add arguments
    parser.add_argument("--grid-size", dest="N", required=False)
    parser.add_argument("--mov-file", dest="movfile", required=False)
    parser.add_argument("--interval", dest="interval", required=False)
    parser.add_argument("--glider", action="store_true", required=False)
    parser.add_argument("--gosper", action="store_true", required=False)
    args = parser.parse_args()

    # # set grid size
    # N = 100
    for N in grid_sizes:
        # print(f"N:{N}")
        t = timer()
        if args.N and int(args.N) > 8:
            N = int(args.N)

        # check if "glider" demo flag is specified
        if args.glider:
            grid = np.zeros(N * N).reshape(N, N)
            addGlider(1, 1, grid)
        elif args.gosper:
            grid = np.zeros(N * N).reshape(N, N)
            addGosperGliderGun(10, 10, grid)
        else:
            # populate grid with random on/off - more off than on
            grid = randomGrid(N)

        for i in range(10):
            grid = update_(grid, N)
        times.append(timer() - t)


# call main
if __name__ == "__main__":
    grid_sizes = [32, 64, 128, 256, 512]
    times = []
    main()

    # plt.plot(range(len(grid_sizes)), times)
    # plt.xticks(range(len(grid_sizes)), labels=grid_sizes)
    # plt.xlabel("Grid Size(N)")
    # plt.ylabel("Time Usgae(s)")
    # plt.show()
