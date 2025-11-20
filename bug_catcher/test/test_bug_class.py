"""Tests for the internal functions of bug.py."""

from bug_catcher import bug as bug
from geometry_msgs.msg import PoseStamped, Quaternion  # , TwistStamped
import numpy as np
import pytest


test_bug = bug.Bug(0, PoseStamped(), bug.Color.red)


@pytest.mark.rostest
def test_quaterion_conversions():
    """Check that quat 0, 0, 0, 1 and 1, 0, 0, 0 return properly."""
    quat0001 = Quaternion()
    quat0001.x, quat0001.y, quat0001.z, quat0001.w = 0, 0, 0, 1
    eu0001 = test_bug._euler_from_quaternion(quat0001)
    print(eu0001)

    quat1000 = Quaternion()
    quat1000.x, quat1000.y, quat1000.z, quat1000.w = 1, 0, 0, 0
    eu1000 = test_bug._euler_from_quaternion(quat1000)
    print(eu1000)

    assert eu0001 == pytest.approx((0, 0, 0))
    assert eu1000 == pytest.approx((np.pi, 0, 0))
