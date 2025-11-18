"""Enables various techniques for picking up a detected HexBug"""

from geometry_msgs.msg import Pose
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

class BugCatcher:
    """
    The class containing various techniques for picking up a detected HexBug

    Subscribers:

    """

    def __init__(self, node: Node):
        """Initialize the BugCatcher."""
        self.node = node

        self.cb_group_1 = MutuallyExclusiveCallbackGroup()

        self.user_node.get_logger().debug('BugCatcher initialization complete')

    async def tracking_pick(self) -> (bool):
        """
        Pick up the bug by tracking its position (no velocity)
        TODO: the pose needs to be allowed to be constantly updated. It should take a
        self.current_bug.pose or something like that that can be updated while the trajectory is
        executing and change the end point

        Returns
        -------
        success (bool): True if the robot gripper thinks it picked up an object
        """
        started_tracking = False
        pounce = False
        # 1. Get trajectory to the bug
        # TODO: write func in mpi that checks cartesian paths to a point, then checks RRT if no cart
        bug_pose = Pose ()  # TODO: update this pose
        # TODO: This tracker here should either ignore the fingers or explicitly set them every loop
        tracking = self.node.mpi.GoTo(bug_pose)

        # If at any poin
        if not tracking:
            # Retry once or twice or something like that?
            started_tracking = False
            pass
        else:
            # wait half a second and make sure tracking is is good the whole time
            if not started_tracking:
                start_time = self.node.get_clock().now()
                started_tracking = True
            # Track for 0.5 seconds, then flip the switch to pounce
            if  self.node.get_clock().now() - start_time >= Duration(nanosec=5*10**8):
                pounce = True
                pass
            if pounce:
                self.node.mpi.CloseGripper

        pass

