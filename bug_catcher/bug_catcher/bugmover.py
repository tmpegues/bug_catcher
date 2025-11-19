"""Enables various techniques for picking up a detected HexBug."""

import asyncio
import math

from geometry_msgs.msg import Point, Pose, Quaternion
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node


class BugMover:
    """
    The class containing various techniques for picking up a detected HexBug.

    Subscribers:

    """

    def __init__(self, node: Node):
        """Initialize the BugMover."""
        self.node = node

        self.cb_group_1 = MutuallyExclusiveCallbackGroup()

        self.node.get_logger().debug('BugCatcher initialization complete')

        self.GRIPPER_OFFSET_Z = 0.01  # 1cm above the bridge surface so we don't crash into it

    # -----------------------------------------------------------------
    # Internal Helper Functions
    # -----------------------------------------------------------------
    def _cal_distance(self, p1: Point, p2: Point):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    # TODO： Decide whether use Pose or Point
    def _cal_travel_time(self, start_pose: Pose, end_pose: Pose) -> float:
        """Calculate the estimated travel time needed for robot to move from start to end."""
        distance = math.sqrt(
            (start_pose.x - end_pose.x) ** 2
            + (start_pose.y - end_pose.y) ** 2
            + (start_pose.z - end_pose.z) ** 2
        )

        ROBOT_SPEED = 0.1  # TODO: Update with actual robot speed
        OVERHEAD_TIME = 1.0  # TODO: Update with actual overhead time

        travel_time = distance / ROBOT_SPEED + OVERHEAD_TIME
        return travel_time

    # -----------------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------------
    async def tracking_pick(self) -> bool:
        """
        Pick up the bug by tracking its current state (no anticipation).

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
        bug_pose = Pose()  # TODO: update this pose
        # TODO: This tracker here should either ignore the fingers or directly set them every loop
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
            if self.node.get_clock().now() - start_time >= Duration(nanosec=5 * 10**8):
                pounce = True
                pass
            if pounce:
                self.node.mpi.CloseGripper

        pass

    async def ambushing_pick(self, bridge_end_position: Point) -> bool:
        """
        Pick up the bug by moving to a bridge position and then closing the gripper.

        Args:
        ----
        bridge_end_position (Point): The coordinate where the bridge ends (drop-off point).

        Returns:
        -------
        success (bool): True if the robot gripper thinks it picked up an object.

        """
        bug_pose = self.node.current_bug.pose.position
        bug_vel = self.node.current_bug_speed

        success, current_robot_pose = self.node.mpi.rs.get_ee_pose()

        ambush_pose = Pose()
        ambush_pose.position.x = bridge_end_position.x
        ambush_pose.position.y = bridge_end_position.y
        ambush_pose.position.z = (
            bridge_end_position.z + self.GRIPPER_OFFSET_Z
        )  # Hover slightly above the bridge height
        ambush_pose.orientation = Quaternion(
            x=1.0, y=0.0, z=0.0, w=0.0
        )  # TODO: Update with correct orientation

        # Time race calculation
        dist_bug_to_bridge = self._cal_distance(bug_pose, bridge_end_position)
        t_bug_arrival = dist_bug_to_bridge / max(bug_vel, 0.01)  # Avoid division by zero

        t_robot_travel = self._cal_travel_time(current_robot_pose.position, ambush_pose.position)

        if t_bug_arrival < t_robot_travel:
            self.node.get_logger().info(
                'Ambush aborted: Robot cannot reach ambush point before bug arrives.'
            )
            return False

        # Move to ambush position
        move_success = self.node.mpi.GoTo(ambush_pose)
        if not move_success:
            self.node.get_logger().info('Ambush failed: Robot could not reach ambush position.')
            return False

        await self.node.mpi.OpenGripper()
        self.logger.info('Trap Set! Waiting for bug...')

        timeout_counter = 0
        MAX_WAIT_CYCLES = 1000  # Avoid infinite loop (e.g., 10 seconds)

        while rclpy.ok() and timeout_counter < MAX_WAIT_CYCLES:
            if self.node.current_bug:
                curr_bug_pos = self.node.current_bug.pose.position
                dist_now = self._get_distance(curr_bug_pos, bridge_end_position)

                if dist_now < self.TRAP_TRIGGER_DISTANCE:
                    self.logger.info(f'Bug in range ({dist_now:.3f}m). SNAP!')
                    break  # Exit loop to close gripper

            timeout_counter += 1

            await asyncio.sleep(0.01)  # 100Hz polling

        if timeout_counter >= MAX_WAIT_CYCLES:
            self.logger.info("Ambush Timed out. Bug didn't arrive?")
            return False

        await self.node.mpi.CloseGripper()
        self.logger.info('Ambush sequence complete.')

        return True
