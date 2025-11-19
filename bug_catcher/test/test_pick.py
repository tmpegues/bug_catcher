import unittest

from bug_catcher_interfaces.srv import Pick
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_testing
import pytest
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


@pytest.mark.rostest
def generate_test_description():
    return LaunchDescription(
        [
            Node(
                package='bug_catcher',
                executable='pick_node',
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestFrankaMove(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('pick_node')
        self.node.pick_service = self.create_service(
            Pick,
            '/pick',
            self.pick_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.num_msg_recv = 0

    def tearDown(self):
        self.node.destroy_node()

    def verifyPose(self):
        pass

    def verifyPlanningFail(self):
        pass

    def pick_callback(self, requuest, response):
        """Logic for pick operation."""
        # TODO: May not need the service call actually. Think its best to hard code these.
