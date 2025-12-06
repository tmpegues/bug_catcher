"""
The script implements the 'catcher_node' within the 'bug_catcher' package.

It acts as the physical execution unit. Since the 'color_detection_node' now
handles coordinate transformation (Camera -> Base), this node simply receives
the world-frame coordinates and executes the motion plan.

Subscribers
-----------
+ /wrist_camera/target_bug (bug_catcher_interfaces.msg.BugInfo):
    Receives the target bug pose ALREADY transformed to the Robot Base Frame.

Services/Actions
----------------
+ /move_group (moveit_msgs.action.MoveGroup): For trajectory planning.
+ /execute_trajectory (moveit_msgs.action.ExecuteTrajectory): For robot movement.

Parameters
----------
+ bug_dimensions (double array): [x, y, z] size of the bug for collision model.
+ grasp_height_z (double): The Z height (meters) relative to base to perform the grasp.
+ loop_execution (bool): If true, the node will reset 'is_busy' to catch multiple bugs.

"""

from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher.planningscene import Obstacle

from bug_catcher_interfaces.msg import BugArray, BugInfo

from moveit_msgs.msg import PlanningScene

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from shape_msgs.msg import SolidPrimitive

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker, MarkerArray


class CatcherNode(Node):
    """Executes pick-and-place operations based on pre-processed vision data."""

    def __init__(self):
        """Initialize the Catcher, connecting to interfaces."""
        super().__init__('catcher_node')

        # Initialize MotionPlanningInterface
        self.mpi = MotionPlanningInterface(self)

        # Declare the filename from the launchfile:
        self.declare_parameter('file', 'objects.yaml')
        filename = self.get_parameter('file').value

        # Load the Scene at Init:
        self.mpi.ps.load_scene(filename)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # =========================================
        # 1. Parameter Declarations
        # =========================================

        # Dimensions of the Hexbug: Length, Width, Height (meters)
        self.declare_parameter('bug_dimensions', [0.045, 0.015, 0.015])
        self.bug_dims = self.get_parameter('bug_dimensions').value

        # The physical height of the table/grasping plane relative to robot base
        # This replaces the hardcoded "0.02"
        self.declare_parameter('grasp_height_z', 0.02)
        self.grasp_z = self.get_parameter('grasp_height_z').value

        # TODO: Continuous execution?
        self.declare_parameter('loop_execution', False)
        self.loop_execution = self.get_parameter('loop_execution').value

        self.declare_parameter('target_bug_name', 'target_bug')
        self.target_bug_name = self.get_parameter('target_bug_name').value

        # =========================================
        # 2. Setup Subscribers
        # =========================================
        self.setup_bug_listener()

        # State flags
        self.is_busy = False

        # Subscription for the bug tracking info:
        self.bug_sub = self.create_subscription(
            BugArray, '/bug_god/bug_array', self.bug_callback, 10
        )

        # PUBLISHERS:
        markerQoS = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.mark_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', markerQoS)
        self.planscene = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Establish the required connections and trackers for updating the planningscene each call.
        # Save the last target bug for removal each update:
        self.last_target_bug = None

        self.get_logger().info('Catcher Node: Ready. Listening to /wrist_camera/target_bug...')

    def setup_bug_listener(self):
        """Set up subscriber for the Wrist Camera pose data."""
        cb_group = MutuallyExclusiveCallbackGroup()

        self.bugs_in_frame_listener = self.create_subscription(
            BugInfo, '/wrist_camera/target_bug', self.bug_callback, 10, callback_group=cb_group
        )

    async def execute_catch_sequence(self):
        """Execute the physical motion sequence to catch the bug."""
        bug = self.target_bug_name

        # Ensure RobotState is receiving data
        while rclpy.ok():
            current_joints = self.mpi.rs.get_angles()
            if current_joints is not None and len(current_joints.name) > 0:
                self.get_logger().info('Robot State connected! Starting sequence.')
                break
            await rclpy.sleep(0.1)  # Wait 100ms

        self.get_logger().info('--- Sequence: Get Ready ---')
        if not await self.mpi.GetReady():
            # If planning fails (e.g. out of reach), we abort and reset busy flag
            self.get_logger().error('Failed to reach Ready pose.')
            self.is_busy = False
            return

        self.get_logger().info('--- Sequence: Open Gripper ---')
        if not await self.mpi.OpenGripper():
            return

        self.get_logger().info(f'--- Sequence: Move Above {bug} ---')
        if not await self.mpi.MoveAboveObject(bug):
            self.get_logger().error('Target out of reach or planning failed.')
            self.is_busy = False
            return

        self.get_logger().info(f'--- Sequence: Move Down to {bug} ---')
        if not await self.mpi.MoveDownToObject(bug):
            self.is_busy = False
            return

        self.get_logger().info('--- Sequence: Close Gripper (Pick) ---')
        if not await self.mpi.CloseGripper(bug):
            self.is_busy = False
            return

        self.get_logger().info('--- Sequence: Lift Up ---')
        if not await self.mpi.LiftOffTable():
            self.is_busy = False
            return

        self.get_logger().info('SUCCESS: Bug Caught!')

    def bug_callback(self, bug_msg):
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
        self.marker_array = MarkerArray()
        self.markers = []

        # Break apart each bug message: is_target: 1-> CollisionObject | 0-> Marker
        for i, bug in enumerate(bug_msg.bugs):
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
                self.markers.append(marker)

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
                marker.lifetime.nanosec = 20000000
                self.markers.append(marker)

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
                self.markers.append(marker)
        # Update Rviz markers for all colored bugs:
        self.marker_array.markers = self.markers
        self.mark_pub.publish(self.marker_array)


def main(args=None):
    """Run the main entry point for the Catcher Node."""
    rclpy.init(args=args)

    catcher_node = CatcherNode()

    # Use MultiThreadedExecutor to allow MoveIt actions (which use callbacks)
    # to run in parallel with the main node logic.
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
