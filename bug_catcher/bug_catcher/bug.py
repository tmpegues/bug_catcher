"""A class to represent a single HexBug."""

from enum import auto, Enum

from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped, Vector3
import numpy as np
from std_msgs.msg import Header


class Color(Enum):
    """
    Color tracker for the bugs.

    TODO: Is using a string fine and enum is unnecessary here?
    """

    red = auto()
    black = auto()


class Bug:
    """A class to represent a single HexBug."""

    def __init__(self, ID: int, pose: PoseStamped, color: Color):
        """
        Initialize the Bug.

        Args:
        ----
        ID (int): a unique id for each bug
        pose (PoseStamped): the pose of the bug at the time of the camera frame
        color (Color): the color of the bug

        """
        self.ID = ID
        self.pose = pose
        self.color = color

        self.vel = TwistStamped()
        self.future_pose = PoseStamped()

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
    def _euler_from_quaternion(self, quat: Quaternion | list):
        """
        Calculate Euler angles from a quaternion.

        Args:
        ----
        quat (Quaternion | list): a quaternion to convert to Euler angles

        Returns
        -------
        rx, py, yz (floats): roll, pitch, yaw angles corresponding to quat

        """
        if type(quat) is list:
            quat = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

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
        Calculate the velocity and future pose of the bug based on the last two camera frames.

        self.vel (TwistStamped) will be updated with the average velocity over the last time step
        self.future_pose (Pose) will be updated with the expected position 1 second in the future

        Args:
        ----
        new_pose (PoseStamped): the new pose of the bug
        old_pose (PoseStamped): the old pose of the bug

        """
        vel = TwistStamped(header=new_pose.header)

        t0 = old_pose.header.stamp.sec + old_pose.header.stamp.nanosec
        tf = new_pose.header.stamp.sec + new_pose.header.stamp.nanosec
        t = float(tf - t0)

        # p for position. below is linear velocity in m/s
        px = (new_pose.pose.position.x - old_pose.pose.position.x) / t
        py = (new_pose.pose.position.y - old_pose.pose.position.y) / t
        pz = (new_pose.pose.position.z - old_pose.pose.position.z) / t
        lin_vel_list = [px, py, pz]
        linear_vel = Vector3(x=px, y=py, z=pz)

        # below is angular displacement (rpy) in radians
        angles_old = self._euler_from_quaternion(old_pose.pose.orientation)
        angles_new = self._euler_from_quaternion(new_pose.pose.orientation)
        angular_disp = [new - old for new, old in zip(angles_new, angles_old)]

        # TMP TODO: Remove or fix the below
        # # q is for quaternion. below is angular velocity in rad/sec
        # qx = (new_pose.pose.orientation.x - old_pose.pose.orientation.x) / t
        # qy = (new_pose.pose.orientation.y - old_pose.pose.orientation.y) / t
        # qz = (new_pose.pose.orientation.z - old_pose.pose.orientation.z) / t
        # qw = (new_pose.pose.orientation.w - old_pose.pose.orientation.w) / t
        # eu_rot = self._euler_from_quaternion([qx, qy, qz, qw])
        # angular_vel = Vector3(x=eu_rot[0], y=eu_rot[1], z=eu_rot[2])
        # vel.twist.angular = angular_vel

        vel.twist.linear = linear_vel
        vel.twist.angular.x, vel.twist.angular.y, vel.twist.angular.z = [
            x / t for x in angular_disp
        ]
        self.vel = vel

        # TMP TODO: add in consideration of angular
        time_step_sec = 1.0
        time_step_nano = 0.0
        new_header = Header()
        new_header.stamp.sec = new_pose.header.stamp.sec + time_step_sec
        new_header.stamp.nanosec = new_pose.header.stamp.nanosec + time_step_nano
        self.future_pose.header = new_header
        self.future_pose.pose.orientation = new_pose.pose.orientation
        fpx = new_pose.pose.position.x + lin_vel_list[0] * time_step_sec + time_step_nano
        fpy = new_pose.pose.position.y + lin_vel_list[1] * time_step_sec + time_step_nano
        fpz = new_pose.pose.position.z + lin_vel_list[2] * time_step_sec + time_step_nano
        self.future_pose.pose.position = Point(x=fpx, y=fpy, z=fpz)
