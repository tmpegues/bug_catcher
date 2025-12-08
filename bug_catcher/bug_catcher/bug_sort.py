"""
The script implements the 'catcher_node' within the 'bug_catcher' package.

It acts as the physical execution unit using a Finite State Machine (FSM).
It receives global awareness from the Sky Cam and precise targeting from the Wrist Cam.

Architecture:
-------------
1. Visualization Stream (Sky Cam): Updates RViz markers and inventory counts.
2. Action Stream (Wrist Cam): Triggers the FSM to transition from IDLE to ACTIVE states.
3. FSM Loop: Handles the sequence of Approach -> Servo -> Pick -> Drop -> Verify.

Subscribers
-----------
+ /wrist_camera/target_bug (bug_catcher_interfaces.msg.BugInfo): High-precision target.
+ /bug_god/bug_array (bug_catcher_interfaces.msg.BugArray): Global inventory & viz.

Parameters
----------
+ bug_dimensions: Size of the bug for collision planning.
+ grasp_height_z: The physical table height z-coordinate.
+ loop_execution: If True, automatically returns to IDLE after a catch.
+ target_bug_name: Name of the collision object in MoveIt.

"""

from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher.planningscene import Obstacle

from bug_catcher_interfaces.msg import BugArray, BugInfo, BasePoseArray

from bug_catcher_interfaces.srv import Sort

from moveit_msgs.msg import PlanningScene

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from shape_msgs.msg import SolidPrimitive

from visualization_msgs.msg import Marker, MarkerArray


class SorterNode(Node):
    """Executes sort-and-place service."""

    def __init__(self):
        """Initialize the Catcher, connecting to interfaces."""
        super().__init__('sort_node')

        # Initialize the Motion Planner:
        self.mpi = MotionPlanningInterface(self)

        # Declare the Parameters:
        self.declare_parameter('grasp_height', 0.05)
        self.grasp_z = self.get_parameter('grasp_height').value

        self.declare_parameter('filename', 'objects.yaml')
        filename = self.get_parameter('filename').value

        # Load static scene objects (table, walls, etc.)
        self.mpi.ps.load_scene(filename)

        # SUBSCRIPTIONS:
        # Subscription for the bug tracking info:
        self.bug_sub = self.create_subscription(
            BugArray, '/bug_god/bug_array', self.publish_markers, 10
        )
        # Subscription to gather and store the poses of the drop locations:
        self.drop_sub = self.create_subscription(
            BasePoseArray, 'drop_locs', self.drop_callback, 10
        )
        self.targ_sub = self.create_subscription(
            BugInfo, 'bug_god/target_bug', self.target_callback, 10
        )

        # PUBLISHERS:
        markerQoS = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', markerQoS
        )
        self.planscene = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # SERVICES:
        self.sort_service = self.create_service(
            Sort, '/sort', self.sort_callback, callback_group=MutuallyExclusiveCallbackGroup()
        )

        # VARIABLES:
        # Save the last target bug for removal each update:
        self.last_target_bug = None
        # Establish the locations for the drop off pads:
        self.drop_locs = {}
        # Keep track of the amount of bugs identified in the planning scene:
        self.numbugs = {
            'blue': None,
            'green': None,
            'yellow': None,
            'orange': None,
            'pink': None,
            'purple': None,
        }
        # Keep track of the current target pose:
        self.target_pose = None

        self.get_logger().info('Sorting Node Initialized. Ready for Pick Initialization')

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def publish_markers(self, bug_msg):
        """
        Update detected bug positions and the planning scene.

        Parameters
        ----------
        bug_msg :
            A list/array of bug detection messages. Each bug contains:
            - id: int8
                The unique identifier for that colored bug.
            - is_target : bool
                True if the bug should be treated as the active collision target.
            - pose : geometry_msgs/PoseStamped (or similar)
                The estimated pose of the bug in the camera/base frame.
            - color : str
                The bug's color label (e.g., 'pink', 'blue', ...).

        """
        # Remove the last target bug if it exists:
        if self.last_target_bug is None:
            pass
        else:
            self.mpi.ps.remove_obstacle(self.last_target_bug)
            self.last_target_bug = None
        # Create an array of markers to build the arena:
        marker_array = MarkerArray()
        markers = []

        # Break apart each bug message: is_target: 1-> CollisionObject | 0-> Marker
        for i, bug in enumerate(bug_msg.bugs):
            # Add the bug to the tracker:
            if self.numbugs[bug.color] is None:
                self.numbugs[bug.color] = 1
            else:
                self.numbugs[bug.color] += 1

            # Check if the bug is the target or not:
            marker = Marker()
            marker.header.frame_id = 'base'
            marker.header.stamp = bug.pose.header.stamp  # The bug gets a time stamp.
            marker.ns = 'bug_markers'
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            if bug.target is True:
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.BOX

                prim.dimensions = [0.03, 0.03, 0.03]

                current_target_bug = Obstacle('target_bug', bug.pose.pose, prim)
                self.mpi.ps.add_obstacle(current_target_bug)
                self.last_target_bug = current_target_bug

                # Make the target Marker larger and black:
                marker.pose.position.x = bug.pose.pose.position.x
                marker.pose.position.y = bug.pose.pose.position.y
                marker.pose.position.z = bug.pose.pose.position.z
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 20000000
                markers.append(marker)
            else:
                # The bug is not a current target, just track it as a colored marker and publish.
                # Set the location of the bug:
                marker.pose.position.x = bug.pose.pose.position.x
                marker.pose.position.y = bug.pose.pose.position.y
                marker.pose.position.z = bug.pose.pose.position.z

                marker.pose.orientation.x = bug.pose.pose.orientation.x
                marker.pose.orientation.y = bug.pose.pose.orientation.y
                marker.pose.orientation.z = bug.pose.pose.orientation.z
                marker.pose.orientation.w = bug.pose.pose.orientation.w
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 40000000
                markers.append(marker)

                # Assign a specific id to each marker based on color identity and color:
                if bug.color == 'pink':
                    marker.color.r = 1.0
                    marker.color.g = 0.75
                    marker.color.b = 0.8
                    marker.color.a = 1.0
                    marker.id = int('1' + str(i))  # Sets a unique number based on color
                elif bug.color == 'green':
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.id = int('2' + str(i))
                elif bug.color == 'blue':
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                    marker.color.a = 1.0
                    marker.id = int('3' + str(i))
                elif bug.color == 'orange':
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.id = int('4' + str(i))
                elif bug.color == 'purple':
                    marker.color.r = 0.5
                    marker.color.g = 0.0
                    marker.color.b = 0.5
                    marker.color.a = 1.0
                    marker.id = int('5' + str(i))
                elif bug.color == 'yellow':
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.id = int('6' + str(i))
                markers.append(marker)
        # Update Rviz markers for all colored bugs:
        marker_array.markers = markers
        self.marker_pub.publish(marker_array)

    def drop_callback(self, drop_msg):
        """
        Update base drop locations for each color bug.

        Args
        ----
            drop_msg:
                A dict of color base locations:
                - color : str
                    The bug's color label (e.g., 'pink', 'blue', ...).
                - pose : geometry_msgs/PoseStamped
                The estimated pose of the bug in the camera/base frame.

        """
        # Break apart the message and store the color and pose in a dictionary:
        for drop_pose in drop_msg.base_poses:
            self.drop_locs[drop_pose.color] = drop_pose.pose

    def target_callback(self, target_msg):
        """Update and set the current target position to catch."""
        # Recieves a BugInfo msg:
        self.target_pose = target_msg.pose.pose
        self.target_pose.position.z = self.grasp_z

    # Will continue to sort a color until that color has been fully sorted out of the arena.
    async def sort_callback(self, request, response):
        """Commands the gripper to go retrieve and move the object."""
        self.get_logger().info(f'Received sort for {request.color}')
        # Flip this value to turn off all downstream plans and executions.
        response.success = True

        # Publish the change in the

        # Continue the pickup process until there are no bugs of that color left:
        while self.numbugs[request.color.data] > 0:
            # Initialize the robot at the Home Position:
            # TODO: Have it aim the end eff up so we can see all the bugs.
            response.success = await self.mpi.GetReady()

            # Begin the Commands to retrieve the objext:
            # Move the arm directly above the object:
            if response.success:
                await self.mpi.MoveAboveObject(self.target_pose.position)

            # Opens the gripper
            if response.success:
                response.success = await self.mpi.OpenGripper()

            # Moves directly downwards until the object is between the grippers
            if response.success:
                response.success = await self.mpi.MoveDownToObject(self.target_pose)

            # Closes the grippers and add the target bug on the end effector:
            if response.success:
                response.success = await self.mpi.CloseGripper('target_bug')

            # Lifts the object slightly off the table
            if response.success:
                response.success = await self.mpi.LiftOffTable()

            # Moves the object to the other side of the obstacle
            if response.success:
                # response.success = await self.mpi.MoveToGoal()
                response.success = await self.mpi.MoveAboveObject(
                    self.drop_locs[request.color.data].position
                )

            # Releases the object and detaches the rectangle
            if response.success:
                response.success = await self.mpi.ReleaseObject(self.target_pose.position)

            # Return the robot to the Home Position:
            if response.success:
                response.success = await self.mpi.GetReady()

        self.get_logger.info(f'All {request.color} bugs are caught! Send another service!')
        return response


def main(args=None):
    """Run the main entry point for the Catcher Node."""
    rclpy.init(args=args)
    catcher_node = SorterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(catcher_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        catcher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
