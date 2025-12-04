# The persistent unique ID for each bug detected in a particular frame
int8 id
# A pose (stamped) for the bug detected in a particular frame
geometry_msgs/PoseStamped pose
# A list of of colors for each bug detected in a particular frame
string color
# Set a marker to establish whether the specific bug is a target.
bool target