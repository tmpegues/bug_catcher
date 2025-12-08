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

from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher_interfaces.srv import Pick
from bug_catcher.bugmover import BugMover
from bug_catcher import bug as bug
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node


class PickNode(Node):
    """Node uses the pick service to pick up objects."""

    def __init__(self):
        """Initialize the PickNode."""
        super().__init__('pick_node')

        # SERVICES:
        self.pick_service = self.create_service(
            Pick, '/pick', self.pick_callback, callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Declare the filename from the launchfile:
        self.declare_parameter('file', 'objects.yaml')
        filename = self.get_parameter('file').value

        # Initialize the MotionPlanningInterface:
        self.mpi = MotionPlanningInterface(self)

        # Load the Scene at Init:
        self.mpi.ps.load_scene(filename)

        self.bm = BugMover(self)

        self.last_traj_time = self.get_clock().now()

        self.get_logger().info('PickNode initialized and ready to receive pick requests.')

    # Call the gripper to go pick up the object:
    async def pick_callback(self, request, response):
        """Commands the gripper to go retrieve and move the object."""
        self.get_logger().info('Received pick')
        # Flip this value to turn off all downstream plans and executions.
        response.success = True

        # Initialize the robot at the Home Position:
        response.success = await self.mpi.GetReady()

        # Begin the Commands to retrieve the objext:
        # Move the arm directly above the object:
        if response.success:
            await self.mpi.MoveAboveObject(request.bug.pose.pose)

        # Opens the gripper
        if response.success:
            response.success = await self.mpi.OpenGripper()

        # Moves directly downwards until the object is between the grippers
        if response.success:
            response.success = await self.mpi.MoveDownToObject(request.bug.pose.pose)

        # Closes the grippers
        if response.success:
            response.success = await self.mpi.CloseGripper(
                request.name
            )  # Nolan TODO: Fix planningscene stuff

        # Lifts the object slightly off the table
        if response.success:
            response.success = await self.mpi.LiftOffTable()

        # Moves the object to the other side of the obstacle
        if response.success:
            # response.success = await self.mpi.MoveToGoal()
            response.success = await self.mpi.MoveAboveObject(request.base.pose)

        # Releases the object and detaches the rectangle
        if response.success:
            response.success = await self.mpi.ReleaseObject(request.name)

        # Return the robot to the Home Position:
        if response.success:
            response.success = await self.mpi.GetReady()

        return response


def main(args=None):
    """Entry point for the Pick Node."""
    rclpy.init(args=args)
    pick_node = PickNode()
    rclpy.spin(pick_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
