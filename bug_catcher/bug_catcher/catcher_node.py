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

import asyncio
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
from rclpy.time import Duration

from shape_msgs.msg import SolidPrimitive

from visualization_msgs.msg import Marker, MarkerArray


OFFSET_X = -0.10
OFFSET_Z = -0.13


class State(Enum):
    """Defines the FSM states for the Catcher Node."""

    IDLE = auto()  # Waiting for a target
    STALKING = auto()  # Stalking and catching the target
    DROPPING = auto()  # Moving to sorting bin and opening gripper
    VERIFYING = auto()  # Checking Sky Cam to confirm success


class CatcherNode(Node):
    """Executes pick-and-place operations using a State Machine."""

    def __init__(self):
        """Initialize the Catcher, connecting to interfaces."""
        super().__init__('catcher_node')

        # 1. Initialize Motion Planners
        self.mpi = MotionPlanningInterface(self)
        self.bm = BugMover(self)

        # 2. Parameters
        self.declare_parameter('bug_dimensions', [0.045, 0.015, 0.015])
        self.bug_dims = self.get_parameter('bug_dimensions').value

        self.declare_parameter('grasp_height_z', 0.05 - OFFSET_Z)
        self.grasp_z = self.get_parameter('grasp_height_z').value

        self.declare_parameter('loop_execution', True)
        self.loop_execution = self.get_parameter('loop_execution').value

        self.declare_parameter('target_bug_name', 'target_bug')
        self.target_bug_name = self.get_parameter('target_bug_name').value

        self.declare_parameter('file', 'objects.yaml')
        filename = self.get_parameter('file').value
        # Load static scene objects (table, walls, etc.)
        self.mpi.ps.load_scene(filename)

        # Subscription for the bug tracking info:
        self.bug_sub = self.create_subscription(
            BugArray, '/bug_god/bug_array', self._publish_markers, 10
        )
        # Subscription to gather and store the poses of the drop locations:
        self.drop_sub = self.create_subscription(
            BasePoseArray, 'drop_locs', self.drop_callback, 10
        )

        # PUBLISHERS:
        markerQoS = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', markerQoS)
        self.planscene = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Establish the required connections and trackers for updating the planningscene each call.
        # Save the last target bug for removal each update:
        self.last_target_bug = None

        # Establish the locations for the drop off pads:
        self.drop_locs = {}

        self.get_logger().info('Catcher Node: Ready. Listening to /wrist_camera/target_bug...')

        # 4. State Machine Variables
        self.state = State.IDLE
        self.current_target_info = None  # Stores the BugInfo of the active target
        self.inventory_counts = {}  # Stores current count of bugs by color
        self.count_before_catch = 0  # Snapshot for verification
        self.action_pending = False  # Lock to prevent overlapping async calls

        self.last_traj_time = self.get_clock().now()

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

        # # Sky Camera (The Observer)
        # self.sky_sub = self.create_subscription(
        #     BugArray, '/bug_god/bug_array', self.sky_observer_callback, 10
        # )

        # # Visualization Publishers
        # markerQoS = QoSProfile(depth=10, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # self.marker_pub = self.create_publisher(
        #     MarkerArray, 'visualization_marker_array', markerQoS
        # )
        # self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Load new gripper TCP frame
        self.new_TCP()

        # 6. Main Control Loop Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Catcher Node Initialized. State: IDLE')

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _publish_markers(self, bug_msg):
        """Visualizes bugs in RViz. Adapted from calibration node."""
        marker_array = MarkerArray()

        for i, bug in enumerate(bug_msg.bugs):
            marker = Marker()
            marker.header.frame_id = 'base'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'bug_markers'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Correctly access nested Pose (BugInfo -> PoseStamped -> Pose)
            marker.pose.position.x = bug.pose.pose.position.x
            marker.pose.position.y = bug.pose.pose.position.y
            marker.pose.position.z = bug.pose.pose.position.z
            marker.pose.orientation = bug.pose.pose.orientation

            # Highlight Target Logic for Visualization
            if bug.target:
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0  # Black for target
            else:
                marker.scale.x = 0.03
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                # Color mapping
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
                elif bug.color == 'orange':
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                elif bug.color == 'purple':
                    marker.color.r = 0.5
                    marker.color.b = 0.5
                else:
                    marker.color.r = 1.0
                    marker.color.g = 1.0  # White/Yellow default

            marker.color.a = 1.0
            marker.lifetime.sec = 0  # Persistent until next update
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

    def new_TCP(self):
        """Attach a new TCP frame for the gripper."""
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [0.13, 0.12]  # height, radius

        fixture_pose = Pose()
        fixture_pose.position.x = OFFSET_X
        fixture_pose.position.y = 0.0
        fixture_pose.position.z = -OFFSET_Z / 2.0
        fixture_pose.orientation.w = 1.0

        fixture_name = 'new_tcp_fixture'
        obs = Obstacle(fixture_name, fixture_pose, prim)

        self.mpi.ps.add_obstacle(obs)
        self.mpi.ps.attach_obstacle(fixture_name)

        self.get_logger().info('New TCP frame configured for gripper.')

    # =========================================================================
    # Callbacks (Input Handling)
    # =========================================================================

    def wrist_target_callback(self, msg):
        """Receives high-priority target data from Wrist Cam / Target Decision."""
        if msg is None:
            return

        if self.state in [State.IDLE, State.STALKING]:
            self.current_target_info = msg
            self.current_target_info.pose.pose.orientation.x = 1.0
            self.current_target_info.pose.pose.orientation.y = 0.0
            self.current_target_info.pose.pose.orientation.z = 0.0
            self.current_target_info.pose.pose.orientation.w = 0.0

            if self.state == State.IDLE:
                self.get_logger().info(f'Target Found ({msg.color}). Start STALKING.')

                # Snapshot inventory count before starting
                self.count_before_catch = self.inventory_counts.get(msg.color, 0)

                self.state = State.STALKING

    # def sky_observer_callback(self, msg):
    #     """Maintains global awareness: updates inventory counts and RViz markers."""
    #     # Update Inventory Counts
    #     temp_counts = {}
    #     for bug in msg.bugs:
    #         c = bug.color
    #         temp_counts[c] = temp_counts.get(c, 0) + 1
    #     self.inventory_counts = temp_counts

    #     # Visualize Scene in RViz
    #     self._publish_markers(msg)

    # =========================================================================
    # Main Control Loop (State Machine)
    # =========================================================================

    async def control_loop(self):
        """Check the current state and execute the corresponding logic."""
        # If an async action is currently running, do not re-enter.
        if self.action_pending:
            return

        try:
            self.action_pending = True  # Lock

            if self.state == State.IDLE:
                # Do nothing, waiting for wrist callback
                # await self.mpi.GetReady()
                pass

            elif self.state == State.STALKING:
                if (self.get_clock().now() - self.last_traj_time) > Duration(seconds=0.01):
                    if self.current_target_info:
                        target_to_send = self.current_target_info
                        print(target_to_send)
                        # target_to_send.pose.pose.position.x += OFFSET_X
                        target_to_send.pose.pose.position.z = self.grasp_z
                        log_msg = (
                            'DEBUG: Calling stalking_pick for '
                            f'{target_to_send.color} at Height Z='
                            f'{target_to_send.pose.pose.position.z:.3f}'
                        )
                        self.get_logger().debug(log_msg)
                        success = await self.bm.stalking_pick(target_to_send)
                        self.last_traj_time = self.get_clock().now()

                        if success:
                            log_msg = (
                                '>>> STALKING SUCCESS: Bug Gripped! Transitioning to DROPPING.'
                            )
                            self.get_logger().info(log_msg)
                            self.state = State.DROPPING
                        else:
                            # Still approaching or planning failed (not due to Obstacle)
                            pass

            elif self.state == State.DROPPING:
                color = self.current_target_info.color
                self.get_logger().info(f'DEBUG: Dropping {color} bug...')
                drop_coords = self.drop_locations.get(color, self.drop_locations['default'])

                drop_pose = Pose()
                drop_pose.position.x = drop_coords[0]
                drop_pose.position.y = drop_coords[1]
                drop_pose.position.z = drop_coords[2]
                drop_pose.orientation.x = 1.0
                drop_pose.orientation.w = 0.0

                await self.mpi.GoTo(drop_pose)
                await self.mpi.OpenGripper()
                self.state = State.VERIFYING

            elif self.state == State.VERIFYING:
                self.get_logger().info('DEBUG: Verifying catch...')
                await self.mpi.GetReady()
                await asyncio.sleep(1.0)

                current_count = self.inventory_counts.get(self.current_target_info.color, 0)
                if current_count < self.count_before_catch:
                    self.get_logger().info('>>> SUCCESS: Count decreased.')
                else:
                    self.get_logger().warn('>>> WARNING: Count did not decrease.')

                self.current_target_info = None
                if self.loop_execution:
                    self.state = State.IDLE
                    self.get_logger().info('Looping: Reset to IDLE.')
                else:
                    self.get_logger().info('Task Complete. Stopping.')

        except RuntimeError as e:
            self.get_logger().error(f'Error in control loop: {e}')
            self.state = State.IDLE  # Reset on error
        finally:
            self.action_pending = False  # Unlock


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
