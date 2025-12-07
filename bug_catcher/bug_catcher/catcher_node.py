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

from enum import Enum, auto

from bug_catcher.bugmover import BugMover
from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher.planningscene import Obstacle

from bug_catcher_interfaces.msg import BugArray, BugInfo, BasePoseArray

from geometry_msgs.msg import Pose

from moveit_msgs.msg import PlanningScene

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from shape_msgs.msg import SolidPrimitive

from visualization_msgs.msg import Marker, MarkerArray


class State(Enum):
    """Defines the FSM states for the Catcher Node."""

    IDLE = auto()  # Waiting for a target
    APPROACHING = auto()  # Moving to coarse location (Pre-grasp)
    SERVOING = auto()  # Fine-tuning alignment using visual servoing
    GRASPING = auto()  # Descending and closing gripper
    DROPPING = auto()  # Moving to sorting bin and opening gripper
    VERIFYING = auto()  # Checking Sky Cam to confirm success


class CatcherNode(Node):
    """Executes pick-and-place operations using a State Machine."""

    def __init__(self):
        """Initialize the Catcher, connecting to interfaces."""
        super().__init__('catcher_node')

        # 1. Initialize Motion Planner
        self.mpi = MotionPlanningInterface(self)
        self.bm = BugMover(self)

        # 2. Parameters
        self.declare_parameter('bug_dimensions', [0.045, 0.015, 0.015])
        self.bug_dims = self.get_parameter('bug_dimensions').value

        self.declare_parameter('grasp_height_z', 0.02)
        self.grasp_z = self.get_parameter('grasp_height_z').value

        self.declare_parameter('loop_execution', True)
        self.loop_execution = self.get_parameter('loop_execution').value

        self.declare_parameter('target_bug_name', 'target_bug')
        self.target_bug_name = self.get_parameter('target_bug_name').value

        self.declare_parameter('file', 'objects.yaml')
        filename = self.get_parameter('file').value
        # Load static scene objects (table, walls, etc.)
        self.mpi.ps.load_scene(filename)

        # 3. Drop-off Configuration (Color -> [x, y, z])
        # TODO: Adjust these coordinates based on ACTUAL SETUP
        self.drop_locations = {
            'red': [0.3, 0.4, 0.2],
            'blue': [0.3, -0.4, 0.2],
            'green': [0.4, 0.4, 0.2],
            'pink': [0.4, -0.4, 0.2],
            'orange': [0.5, 0.0, 0.2],
            'purple': [0.6, 0.0, 0.2],
            'default': [0.3, 0.0, 0.2],  # Fallback
        }

        # Subscription for the bug tracking info:
        self.bug_sub = self.create_subscription(
            BugArray, '/bug_god/bug_array', self.bug_callback, 10
        )
        # Subscription to gather and store the poses of the drop locations:
        self.drop_sub = self.create_subscription(
            BasePoseArray, 'drop_locs', self.drop_callback, 10
        )

        # PUBLISHERS:
        markerQoS = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.mark_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', markerQoS)
        self.planscene = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Establish the required connections and trackers for updating the planningscene each call.
        # Save the last target bug for removal each update:
        self.last_target_bug = None

        # Establish the locations for the drop off pads:
        self.drop_locs = {}

        self.get_logger().info('Catcher Node: Ready. Listening to /wrist_camera/target_bug...')

    def setup_bug_listener(self):
        """Set up subscriber for the Wrist Camera pose data."""
        # 4. State Machine Variables
        self.state = State.IDLE
        self.current_target_info = None  # Stores the BugInfo of the active target
        self.inventory_counts = {}  # Stores current count of bugs by color
        self.count_before_catch = 0  # Snapshot for verification
        self.action_pending = False  # Lock to prevent overlapping async calls
        self.last_target_collision_obj = None

        # 5. Subscribers & Publishers
        cb_group = MutuallyExclusiveCallbackGroup()

        # Wrist Camera (The Trigger)
        self.target_sub = self.create_subscription(
            BugInfo,
            '/wrist_camera/target_bug',
            self.wrist_target_callback,
            10,
            callback_group=cb_group,
        )

        # Sky Camera (The Observer)
        self.sky_sub = self.create_subscription(
            BugArray, '/bug_god/bug_array', self.sky_observer_callback, 10
        )

        # Visualization Publishers
        markerQoS = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(
            MarkerArray, 'visualization_marker_array', markerQoS
        )
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # 6. Main Control Loop Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Catcher Node Initialized. State: IDLE')

    # =========================================================================
    # Callbacks (Input Handling)
    # =========================================================================

    def wrist_target_callback(self, msg):
        """
        Receives high-priority target data from Wrist Cam.

        Only updates the target if IDLE or currently APPROACHING.
        """
        if msg is None:
            return

        if self.state == State.IDLE:
            self.get_logger().info(f'Target Found ({msg.color}). Starting Catch Sequence.')
            self.current_target_info = msg
            self.state = State.APPROACHING

        elif self.state == State.APPROACHING:
            # Allow updating target position on-the-fly for better accuracy
            # while moving to the coarse approach point
            self.current_target_info = msg

    def sky_observer_callback(self, msg):
        """Maintains global awareness: updates inventory counts and RViz markers."""
        # 1. Update Inventory Counts
        temp_counts = {}
        for bug in msg.bugs:
            c = bug.color
            temp_counts[c] = temp_counts.get(c, 0) + 1
        self.inventory_counts = temp_counts

        # 2. Visualize Scene in RViz
        self._publish_markers(msg)

    # =========================================================================
    # Main Control Loop (State Machine)
    # =========================================================================

    async def control_loop(self):
        """Check the current state and execute the corresponding logic."""
        # If an async action (like robot movement) is currently running, do not re-enter.
        if self.action_pending:
            return

        try:
            self.action_pending = True  # Lock

            if self.state == State.IDLE:
                # Do nothing, waiting for wrist callback
                pass

            elif self.state == State.APPROACHING:
                await self._handle_approaching()

            elif self.state == State.SERVOING:
                await self._handle_servoing()

            elif self.state == State.GRASPING:
                await self._handle_grasping()

            elif self.state == State.DROPPING:
                await self._handle_dropping()

            elif self.state == State.VERIFYING:
                await self._handle_verifying()

        except RuntimeError as e:
            self.get_logger().error(f'Error in control loop: {e}')
            self.state = State.IDLE  # Reset on error
        finally:
            self.action_pending = False  # Unlock

    # =========================================================================
    # State Handlers (The Logic)
    # =========================================================================

    async def _handle_approaching(self):
        """State: Move above the object (Coarse alignment)."""
        target = self.current_target_info

        # 1. Add Collision Object to MoveIt
        pose_safe = target.pose.pose
        pose_safe.position.z = self.grasp_z  # Flatten to table height

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = self.bug_dims

        obs = Obstacle(self.target_bug_name, pose_safe, prim)
        self.mpi.ps.add_obstacle(obs)
        self.last_target_collision_obj = obs

        # 2. Move Above
        self.get_logger().info(f'Approaching {target.color} bug...')

        # Ensure RobotState is ready
        if not self._check_robot_connection():
            return

        success = await self.bm.stalking_pick(self.current_target_info)

        if success:
            self.state = State.SERVOING
        else:
            self.get_logger().warn('Approach failed. Retrying or Resetting.')
            self.state = State.IDLE

    async def _handle_servoing(self):
        """State: Use Visual Servoing for fine alignment."""
        self.get_logger().info('Visual Servoing...')

        # TODO: Miguel's function here
        success = await self.bm.stalking_pick(self.current_target_info)

        # Placeholder for now:
        await rclpy.sleep(0.5)
        success = True

        if success:
            self.state = State.GRASPING
        else:
            self.get_logger().warn('Visual Servo failed. Aborting.')
            self.state = State.IDLE

    async def _handle_grasping(self):
        """State: Descend, Pick, and Lift."""
        self.get_logger().info('Grasping...')

        # Record inventory BEFORE we pick it up
        color = self.current_target_info.color
        self.count_before_catch = self.inventory_counts.get(color, 0)

        # Execute Pick Sequence
        bug_name = self.target_bug_name

        await self.bm.stalking_pick(bug_name)
        success = await self.mpi.LiftOffTable()

        if success:
            self.state = State.DROPPING
        else:
            self.get_logger().error('Grasp sequence failed during lift.')
            self.state = State.IDLE

    async def _handle_dropping(self):
        """State: Move to bin and release."""
        color = self.current_target_info.color
        self.get_logger().info(f'Dropping off {color} bug...')

        # Get coords
        drop_coords = self.drop_locations.get(color, self.drop_locations['default'])

        # Create Pose
        drop_pose = Pose()
        drop_pose.position.x = drop_coords[0]
        drop_pose.position.y = drop_coords[1]
        drop_pose.position.z = drop_coords[2]
        # Keep gripper pointing down (approximate quaternion)
        drop_pose.orientation.x = 1.0
        drop_pose.orientation.y = 0.0
        drop_pose.orientation.z = 0.0
        drop_pose.orientation.w = 0.0

        # Move and Drop
        await self.mpi.MoveDownToObject(drop_pose)
        await self.mpi.OpenGripper()

        # Detach object from gripper in MoveIt
        self.mpi.ps.remove_obstacle(self.last_target_collision_obj)

        self.state = State.VERIFYING

    async def _handle_verifying(self):
        """State: Return to Ready and check if bug count decreased."""
        self.get_logger().info('Verifying catch...')

        # Move to Ready to clear view for Sky Cam
        await self.mpi.GetReady()

        # Wait a moment for Sky Cam to update
        await rclpy.sleep(1.0)

        color = self.current_target_info.color
        current_count = self.inventory_counts.get(color, 0)

        if current_count < self.count_before_catch:
            self.get_logger().info('>>> SUCCESS: Bug count decreased. Catch Confirmed.')
        else:
            warn_msg = (
                f'> WARNING: Bug count did not decrease '
                f'({self.count_before_catch} -> {current_count}). '
                'Catch may have failed.'
            )
            self.get_logger().warn(warn_msg)

        # Final Logic
        self.current_target_info = None
        if self.loop_execution:
            self.state = State.IDLE
            self.get_logger().info('Resetting to IDLE for next target.')
        else:
            self.get_logger().info('Loop execution disabled. Stopping.')
            # Stay in VERIFYING or switch to a DONE state to prevent looping

    # =========================================================================
    # Helpers
    # =========================================================================

    def _check_robot_connection(self):
        """Ensure we have robot state before planning."""
        current_joints = self.mpi.rs.get_angles()
        if current_joints is not None and len(current_joints.name) > 0:
            return True
        self.get_logger().warn('Waiting for Robot State...')
        return False

    def _publish_markers(self, bug_msg):
        """Visualizes bugs in RViz. Copied from calibration logic."""
        marker_array = MarkerArray()

        for i, bug in enumerate(bug_msg.bugs):
            marker = Marker()
            marker.header.frame_id = 'base'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'bug_markers'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Correctly access nested Pose
            marker.pose.position.x = bug.pose.pose.position.x
            marker.pose.position.y = bug.pose.pose.position.y
            marker.pose.position.z = bug.pose.pose.position.z
            marker.pose.orientation = bug.pose.pose.orientation

            # Highlight Target
            if bug.target:
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0  # Black
            else:
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                # Simple color map
                if bug.color == 'red':
                    marker.color.r = 1.0
                elif bug.color == 'blue':
                    marker.color.b = 1.0
                elif bug.color == 'green':
                    marker.color.g = 1.0
                elif bug.color == 'pink':
                    marker.color.r = 1.0
                    marker.color.g = 0.75
                    marker.color.b = 0.8
                else:
                    marker.color.r = 1.0
                    marker.color.g = 1.0  # White/Yellow for others

            marker.color.a = 1.0
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)

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


def main(args=None):
    """Run the main entry point for the Catcher Node."""
    rclpy.init(args=args)
    catcher_node = CatcherNode()
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
