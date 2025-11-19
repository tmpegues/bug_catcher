"""Integrate the MotionPlanner, PlanningScene, and RobotState."""

from bug_catcher.motionplanner import MotionPlanner
from bug_catcher.planningscene import PlanningSceneClass
from bug_catcher.robotstate import RobotState
from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node


class MotionPlanningInterface:

    def __init__(self, node: Node):
        """Initialize the MotionPlanningInterface."""
        self.node = node
        # Define the class variables:
        group_name = 'fer_arm'
        base_frame = 'base'
        joint_list = [
            'fer_joint1',
            'fer_joint2',
            'fer_joint3',
            'fer_joint4',
            'fer_joint5',
            'fer_joint6',
            'fer_joint7',
            'fer_finger_joint1',
            'fer_finger_joint2',
        ]
        eef_link = 'fer_hand_tcp'

        # Initialize the classes:
        self.rs = RobotState(
            node, group_name, base_frame, joint_list, eef_link
        )
        self.mp = MotionPlanner(node, self.rs, group_name, eef_link)
        self.ps = PlanningSceneClass(node)

        # Declare Pre-Grasp, Grasp, and Goal Coordinates:
        self.vertical = Quaternion()
        self.vertical.x, self.vertical.w = 1.0, 0.0

        self.pre_grasp_coord = Pose()
        self.pre_grasp_coord.orientation = (
            self.vertical
        )  # Always go to pre-grip with ee vertical
        self.grasp_coord = Pose()
        self.grasp_coord.orientation = (
            self.vertical
        )  # Always grasp with the gripper vertical
        self.goal_coord = Pose()
        self.goal_coord.orientation = (
            self.vertical
        )  # Always grasp with the gripper vertical

    async def GetReady(self):
        """Move the robot to the 'ready' position."""
        success, traj = await self.mp.plan_to_named_config('ready')
        if not success:
            self.node.get_logger().error('Failed to plan to ready position.')
            return False
        else:
            self.node.get_logger().info('Planned to ready position successfully.')

        # Exectute the Plan to Home:
        await self.mp.execute_plan(traj)
        return True

    async def MoveAboveObject(self, obj_name):
        """Move the robot end effector above the object."""
        # Get the coordinates of the object to pick up:
        p = self.ps.obstacles[obj_name].pose.position
        self.pre_grasp_coord.position = type(p)(x=p.x, y=p.y, z=p.z)  # Clone object
        self.pre_grasp_coord.position.z = p.z
        self.pre_grasp_coord.position.z += 0.1
        self.pre_grasp_coord.orientation = self.rs.ee_pose.orientation
        # Flip the orientation to work with MoveIt
        self.pre_grasp_coord.orientation.x = self.vertical.x
        self.pre_grasp_coord.orientation.w = self.vertical.w

        self.node.get_logger().info(
            f'Planning to pre-grasp position: {self.pre_grasp_coord}'
        )

        # Move to pre-grasp position:

        # #################### Begin_Citation [8] ##################
        success, plan = await self.mp.plan_cartesian_path(
            waypoints=[self.pre_grasp_coord]
        )

        if not success or plan is None:
            self.node.get_logger().warn('Plan failed at stage: cart pre-grasp. Attempting RRT.')
            success, plan = await self.mp.plan_to_pose(
                self.pre_grasp_coord.position, self.pre_grasp_coord.orientation
            )
            if not success or plan is None:
                self.node.get_logger().warn('Pre-grasp failed both Cartesian and RRT path.')
                return False

        exec_success = await self.mp.execute_plan(plan)
        if not exec_success:
            self.node.get_logger().warn('Execution failed at stage: pre-grasp')
            return False
        self.node.get_logger().info('Motion to pre-grasp position completed')
        # #################### End_Citation [8] ##################

        return True

    async def OpenGripper(self):
        """Open the grippers of the end effector on the robot."""
        gripper = await self.mp.open_gripper()
        if not gripper:
            self.node.get_logger().warn(
                'Gripper failed at stage: open-gripper before pick'
            )
            return False
        return True

    # Move to object-grasp position:
    async def MoveDownToObject(self, obj_name):
        """Lower the object inbetween the gripper fingers."""
        self.grasp_coord.position = self.ps.obstacles[
            obj_name
        ].pose.position

        self.node.get_logger().info(
            f'Planning to move down to object position: {self.grasp_coord}'
        )
        success, plan = await self.mp.plan_cartesian_path(waypoints=[self.grasp_coord])
        if not success or plan is None:
            self.node.get_logger().warn('Planning failed at stage: motion to object')
            return False
        exec_success = await self.mp.execute_plan(plan)
        if not exec_success:
            self.node.get_logger().warn(
                'Execution failed at stage: \
                                    motion to object'
            )
            return False
        self.node.get_logger().info('motion to object completed')
        return True

    async def CloseGripper(self, obj_name):
        """Close the gripper and attaches a block to the end-effector."""
        # We will attach the block to the end effector:
        self.ps.attach_obstacle(obj_name)
        gripper = await self.mp.close_gripper()
        if not gripper:
            self.node.get_logger().warn(
                'Gripper failed at stage: \
                                    close-gripper at object'
            )
            return False
        self.node.get_logger().info('The block has been picked up!')
        return True

    # Move back to pre-grasp position:
    async def LiftOffTable(self):
        """Lift the object up off the table."""
        self.node.get_logger().info(f'Lifting off the table!: {self.pre_grasp_coord}')
        success, plan = await self.mp.plan_cartesian_path(
            waypoints=[self.pre_grasp_coord]
        )
        if not success or plan is None:
            self.node.get_logger().warn(
                'Planning failed at stage:\
                                    pre-grasp after pick'
            )
            return False
        exec_success = await self.mp.execute_plan(plan)
        if not exec_success:
            self.node.get_logger().warn(
                'Execution failed at stage:\
                                    pre-grasp after pick'
            )
            return False
        self.node.get_logger().info('pre-grasp pose with object completed')
        return True

    # Move back to pre-goal position
    async def MoveToGoal(self):
        """Move to the goal position to drop off the object."""
        # Set the Goal position to be on the other side of the
        # table from the pickup locationl.

        # The goal position is a reflection of the object location across the table
        p = self.pre_grasp_coord.position
        self.goal_coord.position = type(p)(x=p.x, y=p.y, z=p.z)
        self.goal_coord.position.y *= -1
        self.goal_coord.position.z -= 0.1

        self.node.get_logger().info(f'Planning to goal position: {self.goal_coord}')
        success, plan = await self.mp.plan_to_pose(
            self.goal_coord.position, self.goal_coord.orientation
        )
        if not success or plan is None:
            self.node.get_logger().warn(
                'Planning failed at stage:\
                                    motion to pre-goal'
            )
            return False
        exec_success = await self.mp.execute_plan(plan)
        if not exec_success:
            self.node.get_logger().warn(
                'Execution failed at stage:\
                                    motion to pre-goal'
            )
            return False
        self.node.get_logger().info('pre-goal pose with object completed')
        return True

    # Open gripper:
    async def ReleaseObject(self, obj_name):
        """Release the object and detaches the object in Rviz."""
        self.ps.detach_obstacle(obj_name)
        gripper = await self.mp.open_gripper()
        if not gripper:
            self.logger.warn('Gripper failed at stage: open-gripper at goal')
            return False
        self.node.get_logger().info('gripper opened and the goal completed!')

        return True

    # Go to a pose by checking if Cartesian is valid and checking RRT if Cartesian is invalid
    async def GoTo(self, pose: Pose) -> bool:
        """
        Go to a pose by checking if Cartesian is valid and checking RRT if Cartesian is invalid.

        Args:
        ----
        pose (Pose) - the target pose to plan and execute to

        Returns
        -------
        (bool) - True if  plan and execution was successful (doesn't specifiy Cart or RRT)

        """
        success = False
        # 1: check cartesian
        success, plan = await self.mp.plan_cartesian_path(
            waypoints=[pose]
        )
        # 2: if Cartesian failed, check RRT
        if not success or plan is None:
            self.node.get_logger().warn('Plan failed at stage: Cartesian. Attempting RRT.')
            success, plan = await self.mp.plan_to_pose(
                pose.position, pose.orientation
            )
            if not success or plan is None:
                self.node.get_logger().warn('Pre-grasp failed both Cartesian and RRT path.')
                return False
        # 3: If we get to this point, then we have a plan. Execute it.
        exec_success = await self.mp.execute_plan(plan)
        if not exec_success:
            self.node.get_logger().warn('Execution failed in GoTo')
            return False

        return True

    async def GripBug(self):
        """
        Close the gripper on a bug.

        Returns
        -------
        bool: True if the gripper closed. False if some part of the execution failed.

        """
        # I think we should set both a positon and an effort, or just an effort
        # width = 0.005 is the default value we set, but 1 cm seems resonable for the Hexbug size
        gripper = await self.mp.close_gripper(width=0.005)
        if not gripper:
            self.node.get_logger().warn(
                'Gripper failed at stage: GripBug'
            )
            return False
        return True
