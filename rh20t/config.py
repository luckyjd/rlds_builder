import numpy as np

output_dir = "/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_rlds"
root_dir = "/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_unrar_test"

# cam_series_number = ["036422060909",
#                      "038522062288",
#                      "045322071843",
#                      "104122062295",
#                      "104122062823",
#                      "104122063550",
#                      "104422070011",
#                      "f0172289"]
#
# wrist_serial_number = "045322071843"
# third_perspective_serial_number = "036422060909"

train_percent = 0.9  # 90%

# transformed_files = [
#     "force_torque",
#     "force_torque_base",
#     "gripper",
#     "high_freq_data",
#     "joint",
#     "tcp",
#     "tcp_base",
# ]

transformed_default_values = {
    'tcp': {'tcp': np.zeros(7, dtype=np.float32), 'robot_ft': np.zeros(6, dtype=np.float32)},
    'tcp_base': {'tcp': np.zeros(7, dtype=np.float32), 'robot_ft': np.zeros(6, dtype=np.float32)},
    'force_torque': {'zeroed': np.zeros(6, dtype=np.float32), 'raw': np.zeros(6, dtype=np.float32)},
    'force_torque_base': {'zeroed': np.zeros(6, dtype=np.float32), 'raw': np.zeros(6, dtype=np.float32)},
    'gripper': {'gripper_command': np.zeros(3, dtype=np.float32), 'gripper_info': np.zeros(3, dtype=np.float32)},
    'high_freq_data': {'zeroed': np.zeros(6, dtype=np.float32), 'raw': np.zeros(6, dtype=np.float32),
                       'tcp': np.zeros(7, dtype=np.float32)},
    'joint': np.zeros(6, dtype=np.float32),
}

keep_origin_transformed_files = ["gripper", "joint"]

state_action_default = "tcp_base"

task_description_file = "rh20t/task_description.json"

COLOR = "color"
DEPTH = "depth"
