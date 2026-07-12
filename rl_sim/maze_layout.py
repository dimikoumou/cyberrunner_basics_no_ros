"""
Maze geometry as swappable data.

This is a PLACEHOLDER serpentine maze that resembles a labyrinth (start top-left,
snake down to a goal bottom-right, past hazard holes). It is deliberately
data-driven so it can be replaced by an exact trace of the real BRIO board later
without touching the physics or the RL env.

All coordinates are in meters, in the same centered frame the state estimator uses.
"""
import numpy as np
from maze_sim import MazeConfig


def default_maze() -> MazeConfig:
    hx, hy = 0.13, 0.11

    # Four internal horizontal walls, each with a WIDE gap (~0.06 m) on alternating
    # sides, so the only route is a serpentine: right -> down -> left -> down ...
    # Wider gaps keep it a real maze while being threadable by a learned policy.
    walls = np.array([
        [-hx,  0.066,  0.07,  0.066],   # wall 1: gap on the RIGHT (x in [0.07, 0.13])
        [-0.07,  0.022,  hx,  0.022],   # wall 2: gap on the LEFT  (x in [-0.13,-0.07])
        [-hx, -0.022,  0.07, -0.022],   # wall 3: gap on the RIGHT
        [-0.07, -0.066,  hx, -0.066],   # wall 4: gap on the LEFT
    ])

    # A few hazard holes, placed toward the corridor EDGES (near a wall) so a
    # centered trajectory passes safely but a drifting/fast one falls in. Small
    # radius keeps the first version learnable; make them bigger/more numerous
    # once an agent reliably solves this.
    holes = np.array([
        [ 0.00,  0.030],   # row 2, toward the lower wall
        [ 0.00, -0.012],   # row 3, toward the lower wall
        [ 0.00, -0.058],   # row 4, toward the lower wall
    ])

    start = np.array([-0.11, 0.088])

    # Ordered waypoints through the serpentine; the LAST one is the goal.
    checkpoints = np.array([
        [ 0.100,  0.088],   # along top row, toward the right gap
        [ 0.100,  0.044],   # dropped through wall 1 into row 2
        [-0.100,  0.044],   # left along row 2
        [-0.100,  0.000],   # dropped through wall 2 into row 3
        [ 0.100,  0.000],   # right along row 3
        [ 0.100, -0.044],   # dropped through wall 3 into row 4
        [-0.100, -0.044],   # left along row 4
        [-0.100, -0.088],   # dropped through wall 4 into row 5
        [ 0.110, -0.088],   # GOAL: bottom-right
    ])

    return MazeConfig(
        hx=hx, hy=hy,
        walls=walls,
        holes=holes,
        hole_radius=0.006,
        checkpoints=checkpoints,
        start=start,
    )
