"""A class to represent a single HexBug."""

from enum import auto, Enum

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped


class Color(Enum):
    """
    Color tracker for the bugs.

    TODO: Is using a string fine and enum is unnecessary here?
    """

    red = auto()
    black = auto()


class Bug:
    """A class to represent a single HexBug."""

    def __init__(self, id: int, pose: PoseStamped, color: Color):
        """
        Initialize the Bug.

        Args:
        ----
        id (int): a unique id for each bug
        pose (PoseStamped): the pose of the bug at the time of the camera frame
        color (Color): the color of the bug

        """
        self.id = id
        self.pose = pose
        self.color = color

        self.vel = TwistStamped()

    def update(self, new_pose: PoseStamped):
        """
        Update the bug with new pose and velocity.

        Args:
        ----
        new_pose (PoseStamped): the new pose of the bug

        """
        self._calc_vel(new_pose, self.pose)
        self.pose = new_pose

    # This conversion needs to be done because the twist.angular is euler and pose.orient is quat
    def _euler_from_quaternion(self, quat):
        """
        Calculate Euler angles from a quaternion.

        Args:
        ----
        quat (Quaternion): a quaternion to convert to Euler angles

        Returns
        -------
        rx, py, yz (floats): roll, pitch, yaw angles corresponding to quat

        """
        t0 = +2.0 * (quat.w * quat.x + quat.y * quat.z)
        t1 = +1.0 - 2.0 * (quat.x * quat.x + quat.y * quat.y)
        rx = np.arctan2(t0, t1)

        t2 = +2.0 * (quat.w * quat.y - quat.z * quat.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        py = np.arcsin(t2)

        t3 = +2.0 * (quat.w * quat.z + quat.x * quat.y)
        t4 = +1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        yz = np.arctan2(t3, t4)

        return rx, py, yz

    def _calc_vel(self, new_pose: PoseStamped, old_pose: PoseStamped):
        """
        Calculate the velocity of the bug based on the last two camera frames.

        Args:
        ----
        new_pose (PoseStamped): the new pose of the bug
        old_pose (PoseStamped): the old pose of the bug

        """
        vel = TwistStamped()
        vel.header = new_pose.header
        x = new_pose.pose.position.x - old_pose.pose.position.x
        y = new_pose.pose.position.y - old_pose.pose.position.y
        z = new_pose.pose.position.z - old_pose.pose.position.z
        vel.twist.linear.x, vel.twist.linear.y, vel.twist.linear.z = x, y, z

        angles_old = self._euler_from_quaternion(old_pose.pose.orientation)
        angles_new = self._euler_from_quaternion(new_pose.pose.orientation)
        angle_diff = [new - old for new, old in zip(angles_new, angles_old)]
        vel.twist.angular.x, vel.twist.angular.y, vel.twist.angular.z = angle_diff
