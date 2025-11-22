"""Enables various techniques for picking up a detected HexBug."""

import asyncio
import math

from bug_catcher import bug as bug

from geometry_msgs.msg import Point, Pose

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node


class BugMover:
    """Class containins various techniques for picking up a detected HexBug."""

    def __init__(self, node: Node):
        """Initialize the BugMover."""
        self.node = node

        self.cb_group_1 = MutuallyExclusiveCallbackGroup()

        self.node.get_logger().debug('BugCatcher initialization complete')

        self.GRIPPER_OFFSET_Z = 0.01  # 1cm above the bridge surface so we don't crash into it

    # -----------------------------------------------------------------
    # Internal Helper Functions
    # -----------------------------------------------------------------
    def _calc_distance(self, p1: Point, p2: Point):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def _calc_travel_time(self, start_pose: Pose, end_pose: Pose) -> float:
        """Calculate the estimated travel time needed for robot to move from start to end."""
        distance = self._calc_distance(start_pose.position, end_pose.position)

        ROBOT_SPEED = 0.1  # TODO: Update with actual robot speed
        OVERHEAD_TIME = 1.0  # TODO: Update with actual overhead time

        travel_time = distance / ROBOT_SPEED + OVERHEAD_TIME
        return travel_time

    # -----------------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------------
    async def stalking_pick(self, bug: bug.Bug, wrist_cam: bool = False) -> bool:
        """
        Pick up the bug by tracking its current state (no anticipation).

        TMP TODO: the pose needs to be allowed to be constantly updated. It should take a
        self.current_bug.pose or something like that that can be updated while the trajectory is
        executing and change the end point
        This function would benefit from using MoveIt's Servo sub-package. This will take a long
        time for me to learn and is therefore being demoted in priority.

        Ben and Pushkar recommended canceling the action if not complete and you want to change it
            That seems pretty good to me

        Args:
        ----
        bug (bug.Bug): The bug to stalk and pick up
        wrist_cam (bool): True if a wrist camera is available and can be used for timing the grasp

        Returns
        -------
        success (bool): True if the robot gripper thinks it grasped an object

        """
        started_tracking = False
        pounce = False
        # 1. Get trajectory to the bug
        bug_pose = Pose()  # TODO: update this pose
        # TODO: This tracker here should either ignore the fingers or directly set them every loop
        tracking = self.node.mpi.GoTo(bug_pose)

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
                success = self.node.mpi.CloseGripper
                # TMP TODO: break the continuous tracking

        return success

    async def ambushing_pick(self, bridge_end_pose: Pose) -> bool:
        """
        Pick up the bug by moving to a bridge position and then closing the gripper.

        Args
        ----
        bridge_end_pose (Pose): The target pose where the robot should wait to ambush the bug.

        Returns
        -------
        success (bool): True if the ambush sequence completed successfully
            (gripper closed), False otherwise.

        """
        # --- Phase 1: Monitoring Loop ---
        opportunity_found = False
        monitor_counter = 0

        while rclpy.ok():
            # Update data
            bug_pose_msg = self.node.current_bug.pose
            bug_vel = getattr(self.node, 'current_bug_speed', 0.0)

            success, current_robot_pose = self.node.mpi.rs.get_ee_pose()
            # Handle potential Tuple return from RobotState for compatibility
            if isinstance(current_robot_pose, tuple):
                current_robot_pose = current_robot_pose[1]

            # Calculate metrics
            dist_bug_to_goal = self._calc_distance(bug_pose_msg.position, bridge_end_pose.position)

            # --- CONDITIONS ---
            # 1. Is the bug on the bridge? (Passed the Y threshold?)
            is_on_bridge = bug_pose_msg.position.y < self.BRIDGE_ENTRY_THRESHOLD_Y

            # 2. Is the bug moving fast enough to be a sprint?
            is_sprinting = bug_vel > 0.05

            # 3. Is the bug far enough from the goal to give us time?
            is_far_enough = dist_bug_to_goal > 0.1

            if is_on_bridge and is_sprinting and is_far_enough:
                t_bug = dist_bug_to_goal / bug_vel

                # Setup Ambush Pose
                ambush_pose = Pose()
                ambush_pose.position.x = bridge_end_pose.position.x
                ambush_pose.position.y = bridge_end_pose.position.y
                ambush_pose.position.z = bridge_end_pose.position.z + self.GRIPPER_OFFSET_Z
                ambush_pose.orientation = bridge_end_pose.orientation

                t_robot = self._calc_travel_time(current_robot_pose, ambush_pose)

                # 4. Time Race: Can we beat the bug?
                if t_bug > t_robot:
                    self.logger.info(
                        f'>>> OPPORTUNITY! Bug ETA: {t_bug:.2f}s, Robot Needs: {t_robot:.2f}s'
                    )
                    opportunity_found = True
                    break

            monitor_counter += 1
            if monitor_counter % 20 == 0:  # Log every 2 seconds
                self.logger.info(
                    f'Watching... OnBridge: {is_on_bridge}, '
                    f'Dist: {dist_bug_to_goal:.2f}m, Speed: {bug_vel:.2f}m/s'
                )

            await asyncio.sleep(0.1)

        if not opportunity_found:
            return False

        # --- Phase 2: Execution ---
        self.logger.info(f'Moving to Ambush Point: {ambush_pose.position}')
        move_success = await self.node.mpi.GoTo(ambush_pose)

        if not move_success:
            self.logger.error('Failed to move to ambush point!')
            return False

        await self.node.mpi.OpenGripper()
        self.logger.info('TRAP SET! Waiting for contact...')

        # --- Phase 3: Trigger ---
        # Wait for bug to enter the trigger radius
        wait_counter = 0
        max_wait_cycles = 6000  # 60 seconds timeout

        while rclpy.ok() and wait_counter < max_wait_cycles:
            curr_bug = self.node.current_bug.pose.position
            dist = self._calc_distance(curr_bug, bridge_end_pose.position)

            if dist < self.TRAP_TRIGGER_DISTANCE:
                self.logger.info(f'>>> SNAP! Bug in range ({dist:.3f}m).')
                break

            wait_counter += 1
            await asyncio.sleep(0.01)

        if wait_counter >= max_wait_cycles:
            self.logger.warn('Ambush timed out. Bug missed?')
            return False

        # Use the base MotionPlanner close_gripper
        await self.node.mpi.mp.close_gripper()
        self.logger.info('AMBUSH COMPLETE.')
        return True

    async def interdicting_pick(self, bug: bug.Bug, wrist_cam: bool = False) -> bool:
        """
        Pick up the bug by moving to its anticipated future state.

        Args:
        ----
        bug (bug.Bug): The bug to interdict and pick up
        wrist_cam (bool): True if a wrist camera is available and can be used for timing the grasp

        Returns
        -------
        success (bool): True if the robot gripper thinks it grasped an object

        """
        # 1. Future pose is currently 1 second projected in the future. Get the ee there in .75 sec
        # Do not execute this if a cartesian path is not available.
        # TMP TODO: figure out how to set the speed of a cartesian trajectory
        tracking = await self.node.mpi.GoTo(pose=bug.future_pose.pose, cart_only=True)

        # 2. Do not continue if the cartesian path failed
        if not tracking:
            return False
        else:
            # 3.a Close gripper at the time of the future pose if wrist cam is not available
            if not wrist_cam:
                # TMP TODO: How long does it take to close the gripper?
                while self.node.get_clock.now() < bug.future_pose.header.stamp:
                    pass
                self.node.mpi.GripBug()
            elif wrist_cam:
                # While the distance from the bug to the ee pose is too large, don't do anything
                success = False
                while not success:
                    ee_pose = self.node.mpi.rs.get_ee_pose()
                distance = np.linalg.norm(
                    [
                        bug.pose.pose.position.x - ee_pose.pose.position.x,
                        bug.pose.pose.position.y - ee_pose.pose.position.y,
                    ]
                )
                while distance > 0.03:  # 3 cm separation is when we close the grippers
                    pass
                # Once the while loop is exited, close immediately
                self.node.mpi.GripBug()
        return True
