"""Tests for the internal functions of bug.py."""

from bug_catcher import bug as bug
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, TwistStamped, Vector3
import numpy as np
import pytest
from std_msgs.msg import Header


@pytest.mark.rostest
def test_quaterion_conversions():
    """Check that quat 0, 0, 0, 1 and 1, 0, 0, 0 return properly."""
    test_bug = bug.Bug(0, PoseStamped(), bug.Color.red)

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


def test_vel_calculation():
    """Check that velocities are being calculated correctly."""
    # Check a 0, 0, 0 to 1, 1, 1 movement in 1 second gives sqrt(3)/3, sqrt(3)/3, sqrt(3)/3 linear
    header_1s = Header()
    header_1s.stamp.sec, header_1s.stamp.nanosec = 1, 0
    pose_1trans = PoseStamped(header=header_1s, pose=Pose(position=Point(x=1.0, y=1.0, z=1.0)))

    test_bug1 = bug.Bug(0, PoseStamped(), bug.Color.red)
    test_bug1.update(pose_1trans)

    twist_trans = TwistStamped(header=header_1s, twist=Twist(linear=Vector3(x=1, y=1, z=1)))
    # Angular default values are 0. We'll leave them

    # Check that 0, 0, 1, 0 rotation in 1 sec gives a 0, 0, pi angular
    pose_0010rot = PoseStamped(pose=Pose(orientation=Quaternion(x=0.0, y=0.0, z=1.0, w=0.0)))
    pose_0010rot.header = header_1s

    test_bug2 = bug.Bug(0, PoseStamped(), bug.Color.red)
    test_bug2.update(pose_0010rot)

    twist_rot = TwistStamped(header=header_1s, twist=Twist(angular=Vector3(x=0.0, y=0.0, z=np.pi)))

    assert test_bug1.vel == twist_trans
    assert test_bug2.vel == twist_rot


def test_future_pose():
    """Check that the future pose calculation is being calculated correctly."""
    # TMP TODO: Add in consideration or rotation
    # Check a 0, 0, 0 to 1, 1, 1 movement in 1 second gives a 2, 2, 2 expected position
    header_1s = Header()
    header_1s.stamp.sec, header_1s.stamp.nanosec = 1, 0
    pose_1trans = PoseStamped(header=header_1s, pose=Pose(position=Point(x=1.0, y=1.0, z=1.0)))

    test_bug1 = bug.Bug(0, PoseStamped(), bug.Color.red)
    test_bug1.update(pose_1trans)

    header_2s = Header()
    header_2s.stamp.sec, header_2s.stamp.nanosec = 2.0, 0.0
    future_pose = PoseStamped(header=header_2s, pose=Pose(position=Point(x=2.0, y=2.0, z=2.0)))

    assert test_bug1.future_pose == future_pose
