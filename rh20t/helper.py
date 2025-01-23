import os
import glob
import json
import re
import cv2
import numpy as np
from collections import defaultdict
from geometry import quat2euler

from rh20t.config import keep_origin_transformed_files, task_description_file, COLOR, DEPTH, \
    state_action_default, transformed_default_values


def parse_task_from_scene(scene_name):
    """
    get task name from folder name
    """
    match = re.search(r'(task_\d+)', scene_name)
    if match:
        return match.group(1)
    return None


def get_language_info(task_name):
    """
    Read 'task_description.json' (if have) => get all text English & Chinese.
    """
    current_path = os.getcwd()

    json_path = os.path.join(current_path, task_description_file)
    if not os.path.exists(json_path):
        return "", ""
    with open(json_path, "r") as f:
        desc_dict = json.load(f)
    if task_name in desc_dict:
        eng_text = desc_dict[task_name].get("task_description_english", "")
        ch_text = desc_dict[task_name].get("task_description_chinese", "")
        return eng_text, ch_text
    return "", ""


def build_all_cameras(scene_folder):

    cam_dict = {}
    cam_folders = glob.glob(os.path.join(scene_folder, "cam_*"))
    for cfolder in cam_folders:
        cname = os.path.basename(cfolder)
        frames_dict = load_camera_frames_and_timestamps(cfolder)
        if frames_dict:
            cam_dict[cname.replace("cam_", "")] = frames_dict
    return cam_dict


def load_camera_frames_and_timestamps(cam_folder):
    ts_path = os.path.join(cam_folder, "timestamps.npy")
    timestamps = np.load(ts_path, allow_pickle=True)
    ts_lst = timestamps.item()
    return {
        COLOR: load_color_frames_from_cam(cam_folder, ts_lst[COLOR]),
        DEPTH: load_depth_frames_from_cam(cam_folder, ts_lst[DEPTH])
    }


def load_depth_frames_from_cam(cam_folder, timestamps, size=(640, 360)):
    if not timestamps:
        return {}
    frames_dict = {}
    cam_path = os.path.join(cam_folder, f"{DEPTH}.mp4")
    if not os.path.exists(cam_path):
        return {}
    width, height = size
    cap = cv2.VideoCapture(cam_path)

    idx = 0
    is_l515 = ("cam_f" in cam_path)
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(timestamps):
            break
        ts = timestamps[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray1 = np.array(gray[:height, :]).astype(np.int32)
        gray2 = np.array(gray[height:, :]).astype(np.int32)
        gray = np.array(gray2 * 256 + gray1).astype(np.uint16)
        if is_l515:
            gray = gray * 4
        frames_dict[ts] = gray[..., None]
        idx += 1

    cap.release()

    return frames_dict


def load_color_frames_from_cam(cam_folder, timestamps):
    if not timestamps:
        return {}
    frames_dict = {}
    cam_path = os.path.join(cam_folder, f"{COLOR}.mp4")
    if not os.path.exists(cam_path):
        return {}
    cap = cv2.VideoCapture(cam_path)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(timestamps):
            break
        ts = timestamps[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_dict[ts] = frame_rgb
        idx += 1
    cap.release()
    return frames_dict


def build_robot_data(scene_folder):
    transformed_path = os.path.join(scene_folder, "transformed")
    if not os.path.exists(transformed_path):
        return {}
    output = defaultdict()
    for file in list(transformed_default_values.keys()):
        keep_origin_flg = True if file in keep_origin_transformed_files else False
        output[file] = build_transform_data(file, transformed_path, keep_origin_flg)
    return output


def build_transform_data(file_name, transformed_path, keep_origin_flg=False):
    file_path = os.path.join(transformed_path, f"{file_name}.npy")
    if not os.path.exists(file_path):
        return None
    file_data = np.load(file_path, allow_pickle=True).item()

    if keep_origin_flg:
        return file_data
    output = defaultdict()
    for key in list(file_data.keys()):
        data_list = file_data[key]  # [ {"timestamp":..., "tcp":..., ...}, ...]
        out = defaultdict()
        for item in data_list:
            ts = item["timestamp"]

            out[ts] = item
        output[key] = out
    return output


def sync_and_create_episode(serial_number, cam_dict, robot_dict, eng_text, ch_text):
    if not cam_dict or not robot_dict:
        return []

    steps = []
    is_first = True
    for ts in robot_dict[state_action_default][serial_number]:
        state = check_none(robot_dict, state_action_default, serial_number, ts, "tcp")
        metadata = generate_step_metadata(serial_number, ts, robot_dict, ch_text)
        if state is not None:
            euler = quat2euler(state[3:])
            state = np.concatenate((state[:3], euler, np.array([metadata["gripper"]["gripper_info"][0]])))
            state = state.astype(np.float32)
        step = {
            "observation": {
                "image": check_none(cam_dict, serial_number, COLOR, ts),
                # "image_depth": check_none(cam_dict, serial_number, DEPTH, ts),
                # "wrist_image": check_none(cam_dict, serial_number, COLOR, ts),
                # "wrist_image_depth": check_none(cam_dict, serial_number, DEPTH, ts),
                "state": state,
            },
            "action": None,
            "reward": 0.0,
            "discount": 1.0,
            "language_instruction": eng_text,
            # "metadata": metadata,
            "is_first": is_first,
            "is_last": False,
            "is_terminal": False,
        }
        steps.append(step)
        is_first = False

    if steps:
        steps[-1]["is_last"] = True
        steps[-1]["is_terminal"] = True
    return steps


def check_none(data, *keys):
    try:
        value = data
        for key in keys:
            value = value[key]
        return value
    except KeyError:
        return None


def generate_step_metadata(serial_number, timestamp, robot_dict, ch_text):
    metadata = {
                "timestamp": timestamp,
                "task_description_chinese": ch_text,
            }

    for file in list(transformed_default_values.keys()):
        data = check_none(robot_dict, file, serial_number, timestamp)
        if data is not None:
            if isinstance(data, dict):
                data.pop("timestamp", None)
                for key in list(data.keys()):
                    if data[key] is None:
                        data[key] = transformed_default_values[file][key]
                    else:
                        if isinstance(data[key], np.ndarray):
                            data[key] = data[key].astype(np.float32)
            if isinstance(data, np.ndarray):
                data = data.astype(np.float32)
            metadata.update({
                file: data,
            })
        else:
            metadata.update({
                file: transformed_default_values[file]
            })

    return metadata


def postprocess_action_as_next_state(steps):
    for i in range(len(steps) - 1):
        steps[i]["action"] = steps[i + 1]["observation"]["state"]
    # last step => 7 zeros
    steps[-1]["action"] = np.zeros(7, dtype=np.float32)


if __name__ == '__main__':
    scene_name = "task_0001_user_0002_scene_0003_cfg_0001"
    data_root = "/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_unrar"
    task_name = parse_task_from_scene(scene_name)
    # print(get_language_info(data_root, task_name))
    scene_folder = "/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_unrar/RH20T_cfg3/task_0001_user_0016_scene_0001_cfg_0003"
    print(build_all_cameras(scene_folder))
