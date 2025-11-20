"""A class to represent a single HexBug."""

from enum import auto, Enum

from geometry_msgs.msg import PoseStamped, TwistStamped


class Color(Enum):
    """
    Color tracker for the bugs.

    TODO: Is using a string fine?
    """

    red = auto()
    black = auto()


class Bug:
    """A class to represent a single HexBug."""

    def __init__(self, pose: PoseStamped, color: Color):
        """
        Initialize the Bug.

        Args:
        ----
        pose (PoseStamped): the pose of the bug at the time of the camera frame
        color (Color): the color of the bug

        """
        self.pose = pose
        self.color = color

    def update(self, new_pose: PoseStamped):
        """
        Update the bug with new pose and velocity.

        Args:
        ----
        new_pose (PoseStamped): the new pose of the bug

        """
        self._calc_vel(new_pose, self.pose)
        self.pose = new_pose

    def _calc_vel(self, new_pose: PoseStamped, old_pose: PoseStamped):
        """
        Calculate the velocity of the bug based on the last two camera frames.

        Args:
        ----
        new_pose (PoseStamped): the new pose of the bug
        old_pose (PoseStamped): the old pose of the bug

        """
        vel = TwistStamped()
        x = new_pose.pose.position.x - old_pose.pose.position.x
        vel.twist.linear.x = x
        self.vel = vel
