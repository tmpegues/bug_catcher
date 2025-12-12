"""
Node to command the robot to pick up and object and avoid collisions.

This class uses the Motion planning interface to perform the following tasks:
    - Moves the arm directly above the object
    - Opens the gripper
    - Moves directly downwards until the object is between the grippers
    - Closes the grippers
    - Lifts the object slightly off the table
    - Attaches a rectangle (roughly corresponding to the size of the object) to the end-effector
    - Moves the object to the other side of the obstacle
    - Releases the object and detaches the rectangle
"""

from bug_catcher.bugmover import BugMover
from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher_interfaces.msg import BugInfo
import rclpy
from rclpy.node import Node
from rclpy.time import Duration


class TurtlePilot(Node):
    """Node uses the pick service and is also the Node being used to turtle-pilot."""

    def __init__(self):
        """Initialize the TurtlePilot node."""
        super().__init__('turtle_pilot')

        # Declare the filename from the launchfile:
        self.declare_parameter('file', 'objects.yaml')
        filename = self.get_parameter('file').value

        # Initialize the MotionPlanningInterface:
        self.mpi = MotionPlanningInterface(self)

        # Load the Scene at Init:
        self.mpi.ps.load_scene(filename)

        self.bm = BugMover(self)

        self.bug_sub = self.create_subscription(BugInfo, 'bug_info', self.info_cb, 10)

        self.last_traj_time = self.get_clock().now()

        self.get_logger().info('PickNode initialized and ready to receive pick requests.')

    async def info_cb(self, bug_msg):
        """
        Receive BugInfo messages and stalk the bug received.

        Args:
        ----
        bug_msg (BugInfo): The BugInfo message corresponding to the target bug._

        """
        await self.mpi.OpenGripper()
        if (self.get_clock().now() - self.last_traj_time) > Duration(seconds=0.3):
            self.get_logger().debug(
                f'{bug_msg.pose.pose.position} TMP (pick_node): info cb in pick_node triggered: '
            )
            await self.bm.stalking_pick(bug_msg)
            self.last_traj_time = self.get_clock().now()


def main(args=None):
    """Entry point for the TurtlePilot node."""
    rclpy.init(args=args)
    turtle_pilot = TurtlePilot()
    rclpy.spin(turtle_pilot)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
