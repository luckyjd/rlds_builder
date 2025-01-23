# v0.2


# v1.7 

- Single Cam as episode
- use quat2euler to convert quat to euler 
- state = tcp_base[:3] + quat2euler(tcp_base[3:]) + gripper_info[0]   ==> still DOF 7 
- action[episode] = start[episode+1]

# v1.7.2 - minimal version

- Single Cam as episode 
- no image_depth, no metadata
