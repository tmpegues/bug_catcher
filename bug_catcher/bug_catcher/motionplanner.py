"""
A motion-planning interface built on top of MoveIt 2 as a ROS 2 API.

This class receives an external node used for action/service clients, logging, and ROS time.
It provides a unified API for:
    - Joint-space planning
    - Pose-based planning (position/orientation/full pose)
    - Cartesian (straight-line) path generation
    - Planning to named configurations
    - Planning to a set joint configuration
    - Execution of planned trajectories
    - Gripper control (open/close via ExecuteTrajectory)
"""

from os import path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from bug_catcher.robotstate import RobotState
from builtin_interfaces.msg import Duration
from control_msgs.msg import SpeedScalingFactor
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    RobotTrajectory,
)
from moveit_msgs.msg import RobotState as MoveItRobotState
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import xacro


class MotionPlanner:
    """
    An interface for MoveIt2 planning functionalities.

    This class uses an injected Node instance to create its action/service clients.
    This will allow us to connect to RobotState and PlanningScene for the MotionPlanningInterface.

    Subscribers:
    ----------
    user_pose (Pose):  For testing the direct joint control

    Publishers:
    ----------
    /fer_arm_controller/joint_trajectory (JointTrajectory): Allows direct trajectory control,
                                                                bypassing MoveIt

    """

    def __init__(self, node: Node, robot_state: RobotState, group_name: str, eef_link: str):
        """
        Initialize the MotionPlanner.

        Args:
        ----
        node : Node
            External ROS 2 node used to create clients and log events.
        robot_state : RobotState
            Source of the robot's current joint values.
        group_name : str
            Name of the MoveIt planning group.
        eef_link : str
            End-effector link name.

        """
        # Initialize the Instance of the node:
        self.node = node
        # Shortcut for simpiler logging instances:
        self.logger = self.node.get_logger()
        # Definitition of RobotState & Configuration:
        self.robot_state = robot_state
        self.group_name = group_name
        self.eef_link = eef_link
        self.joint_names = [f'fer_joint{i + 1}' for i in range(7)]
        self.gripper_group_name = 'hand'
        self.gripper_joint_names = ['fer_finger_joint1', 'fer_finger_joint2']

        # #################### Begin_Citation [5] ##################
        self.cb_group = ReentrantCallbackGroup()
        # #################### End_Citation [5] ####################

        # CLIENTS:
        # Move Action Client:
        self.move_group_client = ActionClient(
            self.node, MoveGroup, '/move_action', callback_group=self.cb_group
        )
        # Action client for executing saved trajectories
        self.execute_traj_client = ActionClient(
            self.node,
            ExecuteTrajectory,
            '/execute_trajectory',
            callback_group=self.cb_group,
        )
        # Cartesian path planning Client:
        self.cartesian_client = self.node.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=self.cb_group
        )

        self.logger.info('Waiting for MoveIt action and service servers...')

        # Run-Time Errors for Clients:
        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.logger.error('/move_group action server not available.')
            raise RuntimeError('Failed to find /move_group action server.')

        if not self.execute_traj_client.wait_for_server(timeout_sec=5.0):
            self.logger.error('/execute_trajectory action server not available.')
            raise RuntimeError('Failed to find /execute_trajectory action server.')

        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            self.logger.error('/compute_cartesian_path service not available.')
            raise RuntimeError('Failed to find /compute_cartesian_path service.')

        self.logger.info('Parsing SRDF for named configurations...')
        self._group_states = {}
        try:
            xacro_file_path = path.join(
                get_package_share_directory('franka_fer_moveit_config'),
                'srdf',
                'fer_arm.srdf.xacro',
            )
            urdf = xacro.process_file(xacro_file_path).toxml()
            root = ET.fromstring(urdf)

            for group_state in root.findall(f".//group_state[@group='{self.group_name}']"):
                state_name = group_state.get('name')
                joint_values = {}
                for joint in group_state.findall('joint'):
                    joint_values[joint.get('name')] = float(joint.get('value'))

                self._group_states[state_name] = {'joints': joint_values}
            self.logger.info(f'Successfully loaded {len(self._group_states)} named states.')
        except (FileNotFoundError, ET.ParseError, xacro.XacroException) as e:
            self.logger.error(f'Failed to parse SRDF for group states: {e}')

        self.last_plan = None  # To store and inspect the last plan
        self.logger.info('MotionPlanner initialized successfully!')

        self.logger.info('Parsing SRDF for named configurations...')
        self._group_states = {}
        try:
            xacro_file_path = path.join(
                get_package_share_directory('franka_fer_moveit_config'),
                'srdf',
                'fer_arm.srdf.xacro',
            )
            urdf = xacro.process_file(xacro_file_path).toxml()
            root = ET.fromstring(urdf)

            for group_state in root.findall(f".//group_state[@group='{self.group_name}']"):
                state_name = group_state.get('name')
                joint_values = {}
                for joint in group_state.findall('joint'):
                    joint_values[joint.get('name')] = float(joint.get('value'))

                self._group_states[state_name] = {'joints': joint_values}
            self.logger.info(f'Successfully loaded {len(self._group_states)} named states.')
        except (FileNotFoundError, ET.ParseError, xacro.XacroException) as e:
            self.logger.error(f'Failed to parse SRDF for group states: {e}')

        self.last_plan = None  # To store and inspect the last plan
        self.logger.info('MotionPlanner initialized successfully!')

        self.traj_sub = self.node.create_subscription(Pose, 'user_pose', self.user_traj_cb, 10)
        self.direct_traj_pub = self.node.create_publisher(
            JointTrajectory, '/fer_arm_controller/joint_trajectory', 10
        )

    # -----------------------------------------------------------------
    # Internal Helper Functions
    # -----------------------------------------------------------------

    def _get_current_robot_state_msg(self) -> MoveItRobotState:
        """
        Get the current joint state and build a MoveItRobotState message.

        Retrieves the current joint state from self.robot_state and builds
            a MoveItRobotState message.
        """
        rs_msg = MoveItRobotState()

        current_joint_state_msg = self.robot_state.get_angles()

        if (
            current_joint_state_msg is not None
            and current_joint_state_msg.position is not None
            and len(current_joint_state_msg.position) >= len(self.joint_names)
        ):
            rs_msg.joint_state = current_joint_state_msg
        else:
            self.logger.warn(
                'Could not get current joint state. '
                'RobotState.get_angles() is not returning valid data.'
            )
        return rs_msg

    def _create_motion_plan_request(self, start_config=None) -> MotionPlanRequest:
        """Create a generic MotionPlanRequest, taking 20 attempts with 8s plan time."""
        request = MotionPlanRequest()
        request.group_name = self.group_name
        request.planner_id = 'RRTConnectkConfigDefault'
        request.allowed_planning_time = 5.0
        request.num_planning_attempts = 10
        request.workspace_parameters.header.frame_id = 'base'

        if start_config:
            # Provided starting joint configuration
            js = JointState()
            js.name = self.joint_names
            js.position = start_config
            request.start_state.joint_state = js
        else:
            # Use the robot's current position as the start
            request.start_state = self._get_current_robot_state_msg()

        return request

    async def _send_plan_request(self, motion_plan_request: MotionPlanRequest):
        """Send planning request to /move_group action server."""
        goal_msg = MoveGroup.Goal()
        goal_msg.request = motion_plan_request
        goal_msg.planning_options.plan_only = True
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = False

        self.logger.info('Sending plan request to /move_group...')

        try:
            goal_handle_future = self.move_group_client.send_goal_async(goal_msg)
            goal_handle = await goal_handle_future
        except (RuntimeError, ValueError, TypeError, AssertionError) as e:
            # Catch a restricted set of likely exceptions from the
            # action client instead of a blanket `except Exception:`.
            msg = f'Failed to send goal to /move_group: {type(e).__name__}: {e}'
            self.logger.error(msg)
            return False, None

        if not goal_handle.accepted:
            self.logger.error('Planning goal was rejected by /move_group server.')
            return False, None

        self.logger.info('Planning goal accepted, awaiting result...')

        result_response = await goal_handle.get_result_async()

        if result_response.status == 4:  # 4 = 'SUCCEEDED' in ActionStatus
            self.logger.info('Planning successful.')
            self.last_plan = result_response.result.planned_trajectory  # Save
            return True, result_response.result.planned_trajectory
        else:
            err_msg = (
                'Planning failed. '
                f'Status: {result_response.status}, '
                f'Error Code: {result_response.result.error_code.val}'
            )
            self.logger.error(err_msg)
            self.last_plan = None
            return False, None

    # ###################### Begin_Citation [6] ####################
    async def _set_gripper_state(self, positions: list, time_from_start_sec: int = 1):
        """Create and send a 2-POINT trajectory to /execute_trajectory."""
        if not self.execute_traj_client.server_is_ready():
            self.logger.error('/execute_trajectory server not ready.')
            return False

        if len(positions) != len(self.gripper_joint_names):
            self.logger.error(
                f'Gripper command error: expected '
                f'{len(self.gripper_joint_names)} positions, '
                f'but got {len(positions)}.'
            )
            return False

        # Get the robot's CURRENT gripper positions
        current_joint_state_msg = self.robot_state.get_angles()
        if current_joint_state_msg is None:
            self.logger.error('Cannot set gripper state: RobotState is not yet available.')
            return False

        # #################### Begin_Citation [7] ##################
        current_gripper_positions = []
        try:
            for joint_name in self.gripper_joint_names:
                idx = current_joint_state_msg.name.index(joint_name)
                current_gripper_positions.append(current_joint_state_msg.position[idx])
        except ValueError as e:
            self.logger.error(f'Cannot set gripper state: Joint {e} not found in RobotState.')
            return False
        # #################### End_Citation [7] ###################

        # Create the trajectory message
        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = self.gripper_joint_names

        # Create START point (current position)
        start_point = JointTrajectoryPoint()
        start_point.positions = [float(p) for p in current_gripper_positions]
        start_point.time_from_start = Duration(sec=0, nanosec=0)

        # Create GOAL point (target position)
        goal_point = JointTrajectoryPoint()
        goal_point.positions = [float(p) for p in positions]  # The user's goal
        goal_point.time_from_start = Duration(sec=time_from_start_sec, nanosec=0)

        # Add BOTH points
        traj.joint_trajectory.points.append(start_point)
        traj.joint_trajectory.points.append(goal_point)

        # Create Action Goal
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = traj

        self.logger.info(f'Sending gripper command: {self.gripper_joint_names} -> {positions}')

        # Send the goal
        goal_handle = await self.execute_traj_client.send_goal_async(goal_msg)
        if not goal_handle.accepted:
            self.logger.error('Gripper command (ExecuteTrajectory) was rejected.')
            return False

        self.logger.info('Gripper command goal accepted, waiting for result...')
        result_response = await goal_handle.get_result_async()

        if result_response.status == 4:  # SUCCEEDED
            self.logger.info('Gripper command successful.')
            return True
        else:
            self.logger.error(
                f'Gripper command failed. Error code: {result_response.result.error_code.val}'
            )
            return False

    # ###################### End_Citation [6] ######################

    # -----------------------------------------------------------------
    # Public API Functions (Arm Motion Planning)
    # -----------------------------------------------------------------

    async def plan_to_joint_config(self, goal_config: list, start_config: list = None):
        """
        Plan to a target joint configuration [Requirement 1].

        Args
        ----
            goal_config (list): List of target joint angles (float).
            start_config (list, optional): Starting joint angles.
            If None, use current position.

        Returns
        -------
            (bool, RobotTrajectory or None): (success, planned_trajectory)

        """
        self.logger.info(f'Planning to joint config: {goal_config}')
        if len(goal_config) != len(self.joint_names):
            self.logger.error(
                f'Goal config joint count ({len(goal_config)}) '
                f'does not match defined joint count '
                f'({len(self.joint_names)}).'
            )
            return False, None

        request = self._create_motion_plan_request(start_config)

        constraints = Constraints()
        for i, val in enumerate(goal_config):
            jc = JointConstraint()
            jc.joint_name = self.joint_names[i]
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.goal_constraints.append(constraints)
        return await self._send_plan_request(request)

    async def plan_to_pose(
        self,
        goal_position: Point = None,
        goal_orientation: Quaternion = None,
        start_config: list = None,
    ):
        """
        Plan to a target end-effector pose [Requirement 2, 3, 4].

        - If only position is provided, orientation is unconstrained.
        - If only orientation is provided, position is unconstrained.
        - If both are provided, it plans to the full pose.

        Args
        ----
            goal_position (Point, optional): The target position.
            goal_orientation (Quaternion, optional): The target orientation.
            start_config (list, optional): Starting joint angles.
            If None, use current position.

        Returns
        -------
            (bool, RobotTrajectory or None): (success, planned_trajectory)

        """
        if goal_position is None and goal_orientation is None:
            self.logger.error('Must provide at least goal_position or goal_orientation.')
            return False, None

        self.logger.info(f'Planning to pose (Pos: {goal_position}, Orient: {goal_orientation})')
        request = self._create_motion_plan_request(start_config)

        frame_id = 'base'
        constraints = Constraints()

        # Position constraint
        if goal_position is not None:
            pc = PositionConstraint()
            pc.header.frame_id = frame_id
            pc.link_name = self.eef_link

            # Use goal_position, but orientation remains default (unconstrained)
            pose = Pose()
            pose.position = goal_position
            pc.constraint_region.primitive_poses.append(pose)

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.01, 0.01, 0.01]  # 1cm tolerance box
            pc.constraint_region.primitives.append(box)
            pc.weight = 1.0
            constraints.position_constraints.append(pc)

        # Orientation constraint
        if goal_orientation is not None:
            oc = OrientationConstraint()
            oc.header.frame_id = frame_id
            oc.link_name = self.eef_link
            oc.orientation = goal_orientation
            oc.absolute_x_axis_tolerance = 0.01
            oc.absolute_y_axis_tolerance = 0.01
            oc.absolute_z_axis_tolerance = 0.01
            oc.weight = 1.0
            constraints.orientation_constraints.append(oc)

        request.goal_constraints.append(constraints)
        request.max_acceleration_scaling_factor = 0.1
        request.max_velocity_scaling_factor = 0.1
        request.max_cartesian_speed = 0.05

        ##################### Begin_Citation [4] ################## # noqa: E26
        # This constraint prevents the "joint out of limits" ERROR
        # from ros2_control by enforcing it during planning.
        safe_limit_lower = -1.542127
        safe_limit_upper = 1.542127
        jc = JointConstraint()
        jc.joint_name = 'fer_joint7'
        jc.position = (safe_limit_upper + safe_limit_lower) / 2.0  # 0
        jc.tolerance_above = (safe_limit_upper - safe_limit_lower) / 2.0  #
        jc.tolerance_below = (safe_limit_upper - safe_limit_lower) / 2.0
        jc.weight = 1.0

        path_constraints = Constraints()
        path_constraints.joint_constraints.append(jc)
        request.path_constraints = path_constraints
        ##################### End_Citation [4] #################### # noqa: E26

        return await self._send_plan_request(request)

    async def plan_to_position_only(self, goal_position: Point, start_config=None):
        """[Shortcut] Plan to a target end-effector position."""
        return await self.plan_to_pose(goal_position=goal_position, start_config=start_config)

    async def plan_to_orientation_only(self, goal_orientation: Quaternion, start_config=None):
        """[Shortcut] Plan to a target end-effector orientation."""
        return await self.plan_to_pose(
            goal_orientation=goal_orientation, start_config=start_config
        )

    async def plan_to_named_config(self, config_name: str, start_config=None):
        """Plan to a named configuration [Requirement 6]."""
        self.logger.info(f"Planning to SRDF-loaded named config: '{config_name}'")

        if config_name not in self._group_states:
            self.logger.error(
                f"Named config '{config_name}' not found in internal _group_states dictionary."
            )
            return False, None

        goal_config_dict = self._group_states[config_name]['joints']

        try:
            goal_config_list = [goal_config_dict[joint] for joint in self.joint_names]
        except KeyError as e:
            self.logger.error(
                f"Named config '{config_name}' is missing joint '{e}'. Cannot proceed."
            )
            return False, None

        return await self.plan_to_joint_config(goal_config_list, start_config)

    async def plan_cartesian_path(
        self, waypoints: list, start_config: list = None, user_speed: float = 0.0
    ):
        """
        Plan a Cartesian (straight-line) path [Requirement 5].

        Args
        ----
            waypoints (list of Pose): A list of one or more target poses.
            start_config (list, optional): Starting joint angles.
                                           If None, use current position.
            user_speed (float): !=0 will disable the default speed scaling on the trajectory.

        Returns
        -------
            (bool, RobotTrajectory or None): (success, planned_trajectory)

        """
        self.logger.info(f'Planning Cartesian path ({len(waypoints)} waypoints)')
        request = GetCartesianPath.Request()

        request.header.frame_id = 'base'
        request.header.stamp = self.node.get_clock().now().to_msg()

        if start_config:
            js = JointState()
            js.name = self.joint_names
            js.position = start_config
            request.start_state.joint_state = js
        else:
            request.start_state = self._get_current_robot_state_msg()

        request.group_name = self.group_name
        request.link_name = self.eef_link
        request.waypoints = waypoints
        request.avoid_collisions = True
        request.max_step = 0.05
        # request.jump_threshold = 0.1

        ##################### Begin_Citation [4] ################## # noqa: E26
        # This constraint prevents the "joint out of limits" ERROR
        # from ros2_control by enforcing it during planning.
        safe_limit_lower = -1.542127
        safe_limit_upper = 1.542127
        jc = JointConstraint()
        jc.joint_name = 'fer_joint7'
        jc.position = (safe_limit_upper + safe_limit_lower) / 2.0  # 0
        jc.tolerance_above = (safe_limit_upper - safe_limit_lower) / 2.0  #
        jc.tolerance_below = (safe_limit_upper - safe_limit_lower) / 2.0
        jc.weight = 1.0

        path_constraints = Constraints()
        path_constraints.joint_constraints.append(jc)
        request.path_constraints = path_constraints
        ##################### End_Citation [4] #################### # noqa: E26

        if user_speed == 0.0:
            request.max_acceleration_scaling_factor = 0.03
            request.max_velocity_scaling_factor = 0.03
            request.max_cartesian_speed = 0.03
        else:
            request.max_acceleration_scaling_factor = 0.1
            request.max_velocity_scaling_factor = user_speed
            request.max_cartesian_speed = 0.05

        future = self.cartesian_client.call_async(request)
        response = await future

        if (
            response
            and response.solution
            and response.error_code.val == 1
            and response.fraction == 1.0
        ):
            self.logger.info(
                f'Cartesian path planning successful '
                f'(fraction completed: {response.fraction * 100:.2f}%)'
            )
            self.last_plan = response.solution
            return True, response.solution
        else:
            err_val = response.error_code.val if response else 'N/A'
            fraction = response.fraction if response else 0.0
            self.logger.error(
                f'Cartesian path planning failed. '
                f'Error code: {err_val}, '
                f'Fraction: {fraction * 100:.2f}%'
            )
            self.last_plan = None
            return False, None

    # -----------------------------------------------------------------
    # Public API Functions (Gripper Control)
    # -----------------------------------------------------------------

    async def open_gripper(self, width: float = 0.03):
        """
        Commands the gripper to move to an "open" position.

        Args:
        ----
            width (float): Target width (e.g., 0.04 for 4cm).

        Returns
        -------
            bool: True if the command was successful.

        """
        self.logger.info(f"Sending 'open' command (width: {width}m)...")
        positions = [width, width]
        return await self._set_gripper_state(positions)

    async def close_gripper(self, width: float = 0.005, time=1):
        """
        Commands the gripper to move to a "close" position.

        Args:
        ----
            width (float): Target width (e.g., 0.005 for 0.5cm).
           time (float): how long you want closing the gripper to take

        Returns
        -------
            bool: True if the command was successful.

        """
        self.logger.info(f"Sending 'close' command (width: {width}m)...")
        positions = [width, width]
        return await self._set_gripper_state(positions, time_from_start_sec=time)

    # -----------------------------------------------------------------
    # Execution Plan
    # -----------------------------------------------------------------

    async def execute_plan(self, trajectory: RobotTrajectory = None):
        """
        Execute a planned trajectory [Requirement 7].

        Args
        ----
            trajectory (RobotTrajectory, optional): The trajectory to execute.
                If None, it will attempt to execute `self.last_plan`.

        Returns
        -------
            bool: True if execution was successful.

        """
        plan_to_execute = trajectory
        if plan_to_execute is None:
            plan_to_execute = self.last_plan
        if plan_to_execute is None:
            self.logger.warn('No plan available to execute. Call a plan method first.')
            return False

        self.logger.info('Sending execution request to /execute_trajectory...')
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = plan_to_execute

        try:
            goal_handle_future = self.execute_traj_client.send_goal_async(goal_msg)
            goal_handle = await goal_handle_future
        except (RuntimeError, ValueError, TypeError, AssertionError) as e:
            self.logger.error(f'Failed to send execution goal: {type(e).__name__}: {e}')
            return False

        if not goal_handle.accepted:
            self.logger.error('Execution goal was rejected by /execute_trajectory server.')
            return False

        self.logger.info('Execution goal accepted, awaiting result...')

        result_response = await goal_handle.get_result_async()

        if result_response.status == 4:  # SUCCEEDED
            self.logger.info('Trajectory execution successful.')
            return True
        else:
            self.logger.error(
                f'Trajectory execution failed. '
                f'Status: {result_response.status}, '
                f'Error Code: {result_response.result.error_code.val}'
            )
            return False

    def send_direct_traj(self, traj_msg, user_speed: float = 0.0):
        """
        Directly send a trajectory, bypassing MoveIt.

        Args:
        ----
        traj_msg (JointTrajectory): The trajectory you'd like to execute
        user_speed (float): !=0 will allow the user to set their own speed scale to the controller

        """
        if user_speed > 0.0:
            self.direct_speed_pub.publish(SpeedScalingFactor(factor=user_speed))

        self.direct_traj_pub.publish(traj_msg)

    async def user_traj_cb(self, pose_msg: Pose):
        """
        Allow a user to immediately (pending planning) send a trajectory to a Pose.

        Args:
        ----
        pose_msg (Pose): The Poses you'd like to send the robot to

        Returns
        -------
        plan.joint_trajectory (JointTrajectory): The trajectory published to the controller

        """
        self.node.get_logger().info('user request received')
        success = False
        cart_only = False
        # 1: check cartesian
        success, plan = await self.plan_cartesian_path(waypoints=[pose_msg])
        # 2: if Cartesian failed and cart_only isn't set, check RRT
        if cart_only and (not success or plan is None):
            self.node.get_logger().warn('Plan failed at stage: Cartesian. Not Attempting RRT.')
            return False
        elif not success or plan is None:
            self.node.get_logger().warn('Plan failed at stage: Cartesian. Attempting RRT.')
            success, plan = await self.plan_to_pose(pose_msg.position, pose_msg.orientation)
            if not success or plan is None:
                self.node.get_logger().warn('Pre-grasp failed both Cartesian and RRT path.')
                return False
        # 3: If we get to this point, then we have a plan. Execute it.
        if success:
            self.send_direct_traj(plan.joint_trajectory)
