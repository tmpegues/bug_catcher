import unittest

from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher.planningscene import Obstacle
from geometry_msgs.msg import Pose
import launch
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import launch_ros
from launch_ros.substitutions import FindPackageShare
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.time import Duration
from shape_msgs.msg import SolidPrimitive


@pytest.mark.rostest
def generate_test_description():
    moveit_demo_launch = launch.actions.IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('franka_fer_moveit_config'),
                    'launch',
                    'demo.launch.py',
                ]
            ),
        ),
        launch_arguments={'use_rviz': 'False'}.items(),
    )

    # LAUNCH DESCRIPTION - Launches the nodes
    """Launch the nodes under test"""
    return launch.LaunchDescription(
        [
            # Launch your motion planning nodes here
            launch_ros.actions.Node(
                package='moveit_ros_move_group',
                executable='move_group',
                name='move_group',
                output='screen',
                parameters=[{'use_sim_time': False}],
            ),
            moveit_demo_launch,
            # Launch tests after nodes are ready
            launch.actions.TimerAction(
                period=10.0,  # Wait for nodes to initialize
                actions=[launch_testing.actions.ReadyToTest()],
            ),
        ]
    )


# ACTIVE TESTS - Run while nodes are active
class TestMotionPlanningIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # Create the node:
        self.node = rclpy.create_node('test_motion_planning')

        # Set the initial position of the robot to begin testing:
        self.ready_joints = [
            0.0,       # fer_joint1
            -0.785,    # fer_joint2
            0.0,       # fer_joint3
            -2.356,    # fer_joint4
            0.0,       # fer_joint5
            1.571,     # fer_joint6
            0.785      # fer_joint7
        ]

        # Set the position to move to:
        self.target_pose = Pose()
        self.target_pose.position.x = 0.5
        self.target_pose.position.y = 0.0
        self.target_pose.position.z = 0.3
        self.target_pose.orientation.x = 1.0
        self.target_pose.orientation.y = 0.0
        self.target_pose.orientation.z = 0.0
        self.target_pose.orientation.w = 0.0

        # Create an obstacle and set its location within the target pose:
        self.prim = SolidPrimitive()
        self.prim.type = SolidPrimitive.BOX
        self.prim.dimensions = [0.3, 0.3, 0.3]

        # Set the position of the obstacle:
        self.pose = Pose()
        self.pose.position.x = 0.5
        self.pose.position.y = 0.0
        self.pose.position.z = 0.3

    def tearDown(self):
        self.node.destroy_node()

    # ######################### Begin_Citation [10] ####################
    # Unpack the asynchronous call
    def _await(self, coro):
        """Run a coroutine using the ROS global executor."""
        ex = rclpy.get_global_executor()
        future = ex.create_task(coro)
        rclpy.spin_until_future_complete(self.node, future, executor=ex)
        return future.result()
    # ######################### End_Citation [10] ######################

    def test_collision_detection(self):
        """Test that planning fails when target is inside obstacle."""
        # Initialize MPI:
        self.mpi = MotionPlanningInterface(self.node)
        # Wait to establish robot state:
        start = self.node.get_clock().now()
        while (self.node.get_clock().now() - start) < Duration(seconds=3.0):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Initialize and place the obstacle:
        self.obstacle = Obstacle('obstacle', self.pose, self.prim)
        self.mpi.ps.add_obstacle(self.obstacle)

        # Initialize the robot at the Home Position:
        result, plan = self._await(self.mpi.mp.plan_to_joint_config(self.ready_joints))

        # Execute the plan:
        _ = self._await((self.mpi.mp.execute_plan(plan)))

        # Move to collision position:
        result, plan = self._await(
            self.mpi.mp.plan_to_pose(
                self.target_pose.position,
                self.target_pose.orientation
            )
        )

        # Check to make sure that the result is Fail:
        assert not result
