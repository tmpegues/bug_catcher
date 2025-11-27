"""Tests for the internal functions of bugmover.py."""

from bug_catcher import mover_funcs as mf
from geometry_msgs.msg import Point, Pose

import pytest


@pytest.mark.rostest
def test_distance_scaler():
    """Check that 100 cm steps in x, y, z directions are scaled properly (only checks position)."""
    p1 = Pose()
    p2 = Pose(position=Point(x=1))
    p3 = Pose(position=Point(y=1))
    p4 = Pose(position=Point(x=1))
    p5 = Pose(position=Point(x=1, y=1, z=1))

    scale12 = mf._distance_scaler(p1, p2)
    scale13 = mf._distance_scaler(p1, p3)
    scale14 = mf._distance_scaler(p1, p4)
    scale15 = mf._distance_scaler(p1, p5)

    correct12 = Pose(position=Point(x=0.1))
    correct13 = Pose(position=Point(y=0.1))
    correct14 = Pose(position=Point(x=0.1))
    comp_111 = 0.1 * (3**-0.5)
    correct15 = Pose(position=Point(x=comp_111, y=comp_111, z=comp_111))

    pairs = [
        [scale12, correct12],
        [scale13, correct13],
        [scale14, correct14],
        [scale15, correct15],
    ]

    for pair in pairs:
        assert pair[0].position.x == pytest.approx(pair[1].position.x)
        assert pair[0].position.y == pytest.approx(pair[1].position.y)
        assert pair[0].position.z == pytest.approx(pair[1].position.z)
