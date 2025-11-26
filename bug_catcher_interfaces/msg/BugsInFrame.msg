# A list of persistent unique IDs for each bug detected in a particular frame
int8[] id

# A list of poses (stamped) for each bug detected in a particular frame (same order as ids)
geometry_msgs/PoseStamped[] pose

# A list of of colors for each bug detected in a particular frame (same order as ids)
string color