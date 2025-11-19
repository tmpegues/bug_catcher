"""Enables forward and inverse kinematics for the FER and tracks current ee pose and JointState."""

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class RobotState:
    """
    The class for computing kinematics and tracking the robot's state.

    Subscribers:
        + 'joint_states' (sensor_msgs/msg/JointState) - The positions of the FER's joints
        +  tf listener: base_frame -> eef_link

    Server Clients:
        + 'compute_ik' (moveit_msgs/srv/GetPositionIK) - Used to calculate inverse kinematics
        + 'compute_fk' (moveit_msgs/srv/GetPositionFK) - Used to calculate forward kinematics
    """

    def __init__(
        self,
        node: Node,
        group_name: str,
        base_frame: str,
        joint_list: list,
        eef_link: str,
    ):
        """Initialize the RobotState."""
        self.user_node = node
        self.group_name = group_name
        self.base_frame = base_frame
        self.joint_list = joint_list
        self.eef_link = eef_link

        self.ee_pose = Pose()
        self.current_joint_states = JointState()

        self.cb_group_1 = MutuallyExclusiveCallbackGroup()

        self.setup_listeners()
        self.setup_fk_request()
        self.setup_ik_request()

        self.user_node.get_logger().debug('RobotState initialization complete')

    async def ik_callback(self, goal: Pose = None) -> (Int32, JointState):
        """
        Calculate ik for the provided goal.

        Args:
        ----
        goal (Pose): the pose to calculate ik for

        Returns
        -------
        success (Int32): The MoveIt error code for the given pose. 1 = success, -31 = invalid pose.
        resonse (RobotState): A robot state containing a valid JointState to reach the goal

        """
        if not goal:
            goal = self.get_ee_pose()

        # Clear the last requested pose
        self.ik_inquiry.ik_request.pose_stamped = PoseStamped()

        self.stamp_it(self.ik_inquiry.ik_request.pose_stamped)
        self.ik_inquiry.ik_request.pose_stamped.pose = goal
        self.user_node.get_logger().debug('ik_inquiry:')
        self.user_node.get_logger().debug(f'{self.ik_inquiry}')
        self.user_node.get_logger().debug('')

        response = await self.ik_client.call_async(self.ik_inquiry)
        success = response.error_code.val

        self.user_node.get_logger().debug('ik response:')
        self.user_node.get_logger().debug(f'{response}')
        self.user_node.get_logger().debug('')

        self.user_node.get_logger().debug(f'ik success: {success}')

        return success, response

    async def fk_callback(
        self, desired_state: JointState = None
    ) -> (Int32, PoseStamped):
        """
        Calculate fk for the provided goal.

        Args:
        ----
        desired_state (JointState): the pose to calculate fk for

        Returns
        -------
        success (Int32): The MoveItErrorCode from the fk request. 1 = success
        resonse (PoseStamped): The pose that will be reached with the angles at `desired_state`

        """
        if not desired_state:
            desired_state = self.get_angles()

        self.stamp_it(self.fk_inquiry)
        self.user_node.get_logger().debug('fk desired_state:')
        self.user_node.get_logger().debug(f'{desired_state}')
        self.fk_inquiry.robot_state.joint_state.position = desired_state.position
        self.user_node.get_logger().debug('fk_inquiry:')
        self.user_node.get_logger().debug(f'{self.fk_inquiry}')
        self.user_node.get_logger().debug('')

        response = await self.fk_client.call_async(self.fk_inquiry)
        success = response.error_code.val
        self.user_node.get_logger().debug('fk response:')
        self.user_node.get_logger().debug(f'{response}')
        self.user_node.get_logger().debug('')

        self.user_node.get_logger().debug(f'fk success: {success}')

        return success, response

    def get_angles(self) -> JointState:
        """Return the robot's current joint angles."""
        if self.current_joint_states is not None:
            return self.current_joint_states

    def get_ee_pose(self) -> (bool, Pose):
        """
        Get the robot's current ee pose by listening to the tf.

        Returns
        -------
        success (bool): True if tf was retrieved during this callback. False if returning from past
        pose (Pose [not Stamped]): The current (or last) ee position and orientation

        """
        time = rclpy.time.Time()
        try:
            ee_tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.eef_link, time
            ).transform
            self.user_node.get_logger().debug(f'ee_tf: {ee_tf}')
            self.ee_pose.position = ee_tf.translation
            self.ee_pose.orientation = ee_tf.rotation
            return True, self.ee_pose
        except TransformException:
            return False, self.ee_pose

    def joint_callback(self, joint_msg):
        """
        Update the RobotState's joint and finger positions.

        Args:
        ----
        joint_msg (sensor_msgs/msg/JointState): the 'joint_states' message

        """
        self.current_joint_states = joint_msg
        self.ik_inquiry.ik_request.robot_state.joint_state = joint_msg

    def setup_listeners(self):
        """Set up joint_state subscriber and tf_listener."""
        self.joint_listener = self.user_node.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.user_node)

    def setup_ik_request(self):
        """Set up the ik client and message base."""
        # Citation [1]: Andinet was very helpful in confirming the proper format of this request.
        srv_name = '/compute_ik'
        self.ik_client = self.user_node.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.cb_group_1
        )
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'Failed to find GetPositionIK service at {srv_name}')

        self.ik_inquiry = GetPositionIK.Request()
        self.ik_inquiry.ik_request.group_name = self.group_name
        self.ik_inquiry.ik_request.ik_link_name = self.eef_link
        self.ik_inquiry.ik_request.avoid_collisions = True

    def setup_fk_request(self):
        """Set up the fk client and message base."""
        srv_name = '/compute_fk'
        self.fk_client = self.user_node.create_client(
            GetPositionFK, '/compute_fk', callback_group=self.cb_group_1
        )
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'Failed to find GetPositionFK service at {srv_name}')
        self.fk_inquiry = GetPositionFK.Request()
        joints = self.joint_list
        # fk_link_names must be an array, even if only a single link is being calculated for
        self.fk_inquiry.fk_link_names = [self.eef_link]
        self.fk_inquiry.robot_state.joint_state.name = joints

    def stamp_it(self, msg):
        """
        Use this to stamp the header of the argument message with the current time.

        Args:
        ----
        msg (message with a header property): msg.header gets stamped with the current time

        """
        msg.header.stamp = self.user_node.get_clock().now().to_msg()
