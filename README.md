# Mrs. Catfisher Final Project - Bug Sorter
Authors: Halley Zhong, Miguel Pegues, Nolan Knight, Rishika Bera

This package creates a MotionPlanningInterface for the FER Panda robot to pick up and sort colored objects (Turtles!). The robot’s motion planning can be visualized in demo mode or performed in the laboratory using the FER robot to pick up and sort a series of objects while checking for potential collisions with the surrounding environment. The planning scene is fixed for the world that houses the objects, and includes a series of obstacles defined in the Planning Scene. Colored objects are displayed in real time in RViz based on their detected color, and the target 'turtle' is highlighted with a larger black marker.

The MotionPlanningInterface is composed of three subsystems—RobotState, MotionPlanner, and PlanningScene—which interact through MoveIt to control the robot and perform collision detection. In addition to the MotionPlanningInterface, the system includes a TargetDecision node and a Sort node. These use vision-based control to determine the positions of colored objects and assign tasks for the robot to retrieve each item and place it in its designated home position.

The next phase of this project is to sort moving objects! The turtles are powered by Hex-Bugs which allow them to move in realtime, increasing the complexity of the sort task. To support this, we include code and functions for interacting with the bugmover node, which bypasses MoveIt and directly uses the joint-trajectory controller fer_arm_controller. This enables the robot to follow a target pose in real time. If you’re looking to replicate or extend this project, this is a great place to start!

## Quickstart
1. Use `ros2 launch bug_catcher still_turtles.launch.xml demo:=True` to start the Franka demo in Rviz to visualize the demo simulation.
2. Use `ros2 launch bug_catcher still_turtles.launch.xml demo:=False` to start the simulation in the real world on the FER robot. The launch file will only open the visual terminal.
3. Use the service call: `ros2 service call /sort bug_catcher_interfaces/Sort "{color: {data: 'pink'}}"` to command the FER robot to begin sorting the colored objects by a start color.
to the other side of the table.
    NOTE: You will need to connect to you FER robot first and launch its MovIt module before launching the real world launch file and place service call.

3. Here is a video of the robot performing the task in Rviz. This shows how the robot plans to execute each sort operation. The next target bug can be seen being updated during the current pick and sort operation.



https://github.com/user-attachments/assets/768fe2d5-4c71-45b3-9d08-a049a677f79a



4. Here is a video of the robot performing the task in the real world!:


https://github.com/user-attachments/assets/d8dd0a82-1f2d-4ce0-8cf2-a49116a189db


5. Should you want to test communication with the FER using a virtual bug, the standard ROS 2 turtlesim can be used. Use `ros2 launch bug_catcher turtle_pilot.launch.xml turtlesim:=True`. You will need to move the turtlesim turtle yourself: `ros2 run turtlesim turtle_teleop_key' is recommended. The live motion tracking is a work in progress, you will see that it's a tad jerky.

6. If you would like to create and rosdoc2 API Documentation for this package:
    Use `rosdoc2 default_config --package-path bug_catcher`
    Then open the generated html documentation in docs_output. This documentation will give you an overview of each class and Node's inputs, functions, and outputs.
