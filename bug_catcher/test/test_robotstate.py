"""Unit test to verify that the MPI can move the robot from one position to another."""
import asyncio
import unittest

from bug_catcher.motionplanninginterface import MotionPlanningInterface
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


@pytest.mark.rostest
def generate_test_description():
    moveit_demo_launch = launch.actions.IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('franka_fer_moveit_config'),
                'launch',
                'demo.launch.py'
            ]),
        ),  launch_arguments={'use_rviz': 'False'}.items()
    )

    # LAUNCH DESCRIPTION - Launches the nodes
    """Launch the nodes under test"""
    return launch.LaunchDescription(
        [
            # Launch your motion planning nodes:
            launch_ros.actions.Node(
                package='moveit_ros_move_group',
                executable='move_group',
                name='move_group',
                output='screen',
                parameters=[{'use_sim_time': False}],
            ),
            moveit_demo_launch,
            # Launch tests:
            launch.actions.TimerAction(
                period=10.0,  # Waits for nodes to initialize
                actions=[launch_testing.actions.ReadyToTest()],
            ),
        ]
    )


# ACTIVE TESTS - Run while nodes are active
class TestRobotState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_motion_planning')

        # Set the First pose for the robot to move to:
        self.pose1 = Pose()
        self.pose1.position.x = 0.5
        self.pose1.position.y = 0.0
        self.pose1.position.z = 0.5
        self.pose1.orientation.x = 1.0
        self.pose1.orientation.y = 0.0
        self.pose1.orientation.z = 0.0
        self.pose1.orientation.w = 0.0

        # Set the Second pose for the robot to move to:
        self.pose2 = Pose()
        self.pose2.position.x = 0.2
        self.pose2.position.y = 0.2
        self.pose2.position.z = 0.2
        self.pose2.orientation.x = 1.0
        self.pose2.orientation.y = 0.0
        self.pose2.orientation.z = 0.0
        self.pose2.orientation.w = 0.0

    def tearDown(self):
        self.node.destroy_node()

    def poses_are_equal(self, p1, p2, tol=1e-4):
        return (
            abs(p1.position.x - p2.position.x) < tol
            and abs(p1.position.y - p2.position.y) < tol
            and abs(p1.position.z - p2.position.z) < tol
            and abs(p1.orientation.x - p2.orientation.x) < tol
            and abs(p1.orientation.y - p2.orientation.y) < tol
            and abs(p1.orientation.z - p2.orientation.z) < tol
            and abs(p1.orientation.w - p2.orientation.w) < tol
        )


def test_moveToPose(self):
    # Wait to establish robot state:
    start = self.node.get_clock().now()
    while (self.node.get_clock().now() - start) < Duration(seconds=3.0):
        rclpy.spin_once(self.node, timeout_sec=0.1)

    # Initialize MPI:
    self.mpi = MotionPlanningInterface(self.node)

    # Move to first position:
    # ######################## Begin_Citation [9] ######################
    future = asyncio.run(
        self.mpi.mp.plan_to_pose(self.pose1.position, self.pose1.orientation)
    )
    # ######################## End_Citation [9] ######################

    rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
    result, plan = future.result()

    # Execute the plan:
    exec_future = asyncio.run(self.mpi.mp.execute_plan(plan))
    rclpy.spin_until_future_complete(self.node, exec_future, timeout_sec=20.0)

    # Check to make sure we are at position 1:
    _, r1Pose = self.mpi.rs.get_ee_pose()
    assert self.poses_are_equal(self.pose1, r1Pose), 'Position 1 mismatch'

    # Move to second position:
    future = asyncio.run(
        self.mpi.mp.plan_to_pose(self.pose2.position, self.pose2.orientation)
    )
    rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
    result, plan = future.result()

    # Execute the plan:
    exec_future = asyncio.run(self.mpi.mp.execute_plan(plan))
    rclpy.spin_until_future_complete(self.node, exec_future, timeout_sec=20.0)

    # Check to make sure we are at position 2:
    _, r2Pose = self.mpi.rs.get_ee_pose()
    assert self.poses_are_equal(self.pose2, r2Pose), 'Position 2 mismatch'
