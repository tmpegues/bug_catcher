"""Enables various techniques for picking up a detected HexBug."""

import asyncio

from bug_catcher import bug as bug
from bug_catcher import mover_funcs as mv

from geometry_msgs.msg import Pose

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node


class BugMover:
    """Class containins various techniques for picking up a detected HexBug."""

    def __init__(self, node: Node):
        """Initialize the BugMover."""
        self.node = node

        self.GRIPPER_OFFSET_Z = 0.01  # 1cm above the bridge surface so we don't crash into it

        self.cb_group_1 = MutuallyExclusiveCallbackGroup()

        self.last_traj = False
        self.last_waypoints = False  # Save the last trajectory so that we can start interruption
        # paths with the end of the existing path

        self.ee_cup_x_offset = +0.1
        self.ee_cup_z_offset = +0.13

        self.node.get_logger().debug('BugCatcher initialization complete')

    # -----------------------------------------------------------------
    # Public Functions
    # -----------------------------------------------------------------
    async def stalking_pick(self, buginfo_msg, cart_only: bool = True) -> bool:
        """
        Pick up the bug by tracking its current state (no anticipation).

        Args:
        ----
        buginfo_msg (BugInfo): The BugInfo msg for the bug to be picked up
        cart_only (bool): True will restrict the function to only use cartesian paths

        Returns
        -------
        success (bool): True if  gripper successfully closed (False might mean we're still moving)

        """
        user_speed = 1.0

        ee_frame = 'fer_hand_tcp'
        self.node.get_logger().debug(f'{buginfo_msg.pose.pose.position} TMP (bm) bug pose: ')
        self.node.get_logger().debug(
            f'{self.node.mpi.rs.get_ee_pose(frame=ee_frame)[1].position} TMP (bm) current pose: '
        )
        dist_to_bug = mv._calc_distance(
            buginfo_msg.pose.pose,
            self.node.mpi.rs.get_ee_pose(frame=ee_frame)[1],  # TMP TODO: update frame
        )
        success = False
        self.node.get_logger().info(f'Stalking: {dist_to_bug}')

        goal_pose = Pose(orientation=buginfo_msg.pose.pose.orientation)
        goal_pose.position.x = buginfo_msg.pose.pose.position.x + self.ee_cup_x_offset
        goal_pose.position.y = buginfo_msg.pose.pose.position.y
        goal_pose.position.z = buginfo_msg.pose.pose.position.z + self.ee_cup_z_offset
        # While far from the bug, keep moving and do not close the gripper
        if dist_to_bug > 0.10:  # Raise the goal pose if far from the bug
            self.node.get_logger().info('Stalking: Big')

            goal_pose.position.z += 0.05
        elif dist_to_bug > 0.01:  # If close, lower the gripper
            self.node.get_logger().info('Stalking: Mid')
            goal_pose.position = buginfo_msg.pose.pose.position
        elif dist_to_bug < 0.01:  # If really close, close the gripper
            self.node.get_logger().info('Stalking: Small')
            return await self.node.mpi.GripBug()

        self.node.get_logger().info(f'Stalking: goal {goal_pose.position.z}')
        self.node.get_logger().info(f'Stalking: bug {buginfo_msg.pose.pose.position.z}')

        if type(self.last_waypoints) is bool:
            start_pose = self.node.mpi.rs.get_ee_pose(frame=ee_frame)[1]
            start_traj_point = None
        else:
            start_pose = self.last_waypoints[-2]

        if type(self.last_traj) is not bool:
            start_traj_point = self.last_traj.points[-1]
            # start_traj_point = None
            # Start current trajectory from 2nd last waypoint
            # that's being executed
        else:
            start_traj_point = None

        # Generate a Pose path of some number of waypoints
        waypoints = mv.waypoint_maker(start_pose, goal_pose, steps=15)

        # Scale speed by distance if user_speed is True
        if user_speed == 1.0:
            user_speed = max(dist_to_bug, 0.07)

        if type(waypoints) is not bool:
            self.last_waypoints = waypoints
            self.last_traj = await self.node.mpi.interruptable_pose_traj(
                waypoints,
                first_traj_point=start_traj_point,
                cart_only=cart_only,
                user_speed=user_speed,
            )

        return success

        # While more than 1 cm away, keep moving towards the bug
        # Right here, I'm doing discrete 10 cm steps towards the bug until we get close enough.
        #  That should work, as long as the arm is faster than the bug, right?

        # dist_to_bug = mv._calc_distance(
        #     bug.pose.pose,
        #     self.node.mpi.rs.get_ee_pose(frame=ee_frame),  # TMP TODO: update frame
        # )
        # success = True

        # while dist_to_bug >= 0.01 and success is True:
        #     step_pose = mv._distance_scaler(p1=self.node.mpi.rs.get_ee_pose(), p2=bug.pose.pose)
        #     # While the ee pose is more than 3 cm away from the x,y coordinates of the bug,
        #     # follow with the ee raised by 5 cm
        #     if np.linalg.norm([step_pose.position.x, step_pose.position.y]) >= 0.05:
        #         step_pose.position.z += 0.05  # lift by 5 cm
        #     success = await self.node.mpi.GoTo(step_pose, cart_only=True)

        #     dist_to_bug = mv._calc_distance(
        #         bug.pose.pose, self.node.mpi.rs.get_ee_pose(frame=ee_frame)
        #     )
        # if success is False:
        #     self.node.get_logger().info('Stalking pick has failed')
        # else:
        #     success = self.node.mpi.CloseGripper()

        # return success

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
            dist_bug_to_goal = mv._calc_distance(bug_pose_msg.position, bridge_end_pose.position)

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

                t_robot = mv._calc_travel_time(current_robot_pose, ambush_pose)

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
            dist = mv._calc_distance(curr_bug, bridge_end_pose.position)

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
