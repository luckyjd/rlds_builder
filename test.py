import numpy as np
import glob
import os
from collections import defaultdict, Counter

def cal_gap(list_ts):
    for index, item in enumerate(list_ts):
        if index > 0:
            print(list_ts[index] - list_ts[index-1])

if __name__ == '__main__':
    # rh20t = np.load("RH20T_cfg5/calib/1640002326600/tcp.npy", allow_pickle=True)
    folder = "/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_unrar/RH20T_cfg3"
    task = "task_0001_user_0016_scene_0001_cfg_0003"
    #
    # list_folder_all = [
    #     path for path in glob.glob(os.path.join(folder, "task_*_user_*_scene_*_cfg_*/cam_*"))
    #     if "human" not in path
    # ]
    #
    # # Extract just the cam_* part (basename)
    # cam_names = [os.path.basename(path) for path in list_folder_all]
    #
    # # Count occurrences
    # cam_counts = Counter(cam_names)
    #
    # # Print results
    # for cam, count in cam_counts.items():
    #     print(f"{cam}: {count}")
    #
    # # If you need the total number of unique cam_*
    # print(f"Total unique task folders: {len([path for path in glob.glob(os.path.join(folder, 'task_*_user_*_scene_*_cfg_*')) if 'human' not in path])}")
    # # # 1575 vs 799

    # # wrist
    cam_036422060909 = np.load(f"{folder}/{task}/cam_036422060909/timestamps.npy", allow_pickle=True)
    cam_038522062288 = np.load(f"{folder}/{task}/cam_038522062288/timestamps.npy", allow_pickle=True)
    cam_045322071843 = np.load(f"{folder}/{task}/cam_045322071843/timestamps.npy", allow_pickle=True)
    cam_104122062295 = np.load(f"{folder}/{task}/cam_104122062295/timestamps.npy", allow_pickle=True)
    cam_104122062823 = np.load(f"{folder}/{task}/cam_104122062823/timestamps.npy", allow_pickle=True)
    cam_104122063550 = np.load(f"{folder}/{task}/cam_104122063550/timestamps.npy", allow_pickle=True)
    cam_104422070011 = np.load(f"{folder}/{task}/cam_104422070011/timestamps.npy", allow_pickle=True)
    cam_f0172289 = np.load(f"{folder}/{task}/cam_f0172289/timestamps.npy", allow_pickle=True)
    data = np.load(f"{folder}/{task}/transformed/tcp_base.npy", allow_pickle=True)
    print("A")
    #
    # # {"finish_time": 1631270670081, "rating": 9, "calib_quality": 2, "calib": 1631153393825, "action": 1631241424515}
    #
    #
    #
    # print(f"LEN cam_036422060909 : {len(cam_036422060909.item()['color'])}")
    # print(f"START cam_036422060909 : {cam_036422060909.item()['color'][0]}")
    # print(f"END cam_036422060909 : {cam_036422060909.item()['color'][-1]}")
    # print("\n")
    # # print(f"LEN cam_036422060909 : {len(cam_038522062288.item()['color'])}")
    # # print(f"START cam_038522062288 : {cam_038522062288.item()['color'][0]}")
    # # print(f"END cam_038522062288 : {cam_038522062288.item()['color'][-1]}")
    # print("\n")
    # print(f"LEN cam_045322071843 : {len(cam_045322071843.item()['color'])}")
    # print(f"START cam_045322071843 : {cam_045322071843.item()['color'][0]}")
    # print(f"END cam_045322071843 : {cam_045322071843.item()['color'][-1]}")
    # print("\n")
    # print(f"LEN cam_104122062295 : {len(cam_104122062295.item()['color'])}")
    # print(f"START cam_104122062295 : {cam_104122062295.item()['color'][0]}")
    # print(f"END cam_104122062295 : {cam_104122062295.item()['color'][-1]}")
    # print("\n")
    # print(f"LEN cam_f0172289 : {len(cam_f0172289.item()['color'])}")
    # print(f"START cam_f0172289 : {cam_f0172289.item()['color'][0]}")
    # print(f"END cam_f0172289 : {cam_f0172289.item()['color'][-1]}")
    #
    # print(cal_gap(cam_036422060909.item()['color']))


