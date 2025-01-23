import os
import glob
import json
import numpy as np

import tensorflow_datasets as tfds
from rh20t.helper import parse_task_from_scene, get_language_info, build_all_cameras, load_camera_frames_and_timestamps, build_robot_data, sync_and_create_episode, postprocess_action_as_next_state

from rh20t.config import train_percent

_DESCRIPTION = """
RH20T dataset: Robot manipulation raw data -> TFDS.
"""
_CITATION = """
@misc{rh20t_dataset,
  title={RH20T dataset},
  author={...},
  year={2025},
}
"""


class Rh20tDataset(tfds.core.GeneratorBasedBuilder):
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Prepare data and save to:
          --manual_dir=/path/to/RH20T_root
        Structure: RH20T_cfg1, RH20T_cfg2, ...
        """

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        # 'image_depth': tfds.features.Image(
                        #     shape=(360, 640, 1),
                        #     dtype=np.uint16,
                        #     encoding_format='png',
                        #     doc='Depth camera RGB observation.',
                        # ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, xyz+quat (7D) Gripper Cartesian poses, default transformed/tcp_base.npy',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot state, xyz+quat (7D) Gripper Cartesian poses, '
                            'default transformed/tcp_base.npy of next frame',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    # 'metadata': tfds.features.FeaturesDict({
                    #     'task_description_chinese': tfds.features.Text(
                    #         doc='Language Instruction in Chinese.'
                    #     ),
                    #     'timestamp': tfds.features.Scalar(
                    #         dtype=np.float32,
                    #     ),
                    #     'tcp': tfds.features.FeaturesDict({
                    #         'tcp': tfds.features.Tensor(
                    #             shape=(7,),     # 7
                    #             dtype=np.float32,
                    #         ),
                    #         'robot_ft': tfds.features.Tensor(
                    #             shape=(6,),     # 6
                    #             dtype=np.float32,
                    #         ),
                    #     }),
                    #     'tcp_base': tfds.features.FeaturesDict({
                    #         'tcp': tfds.features.Tensor(
                    #             shape=(7,),     # 7
                    #             dtype=np.float32,
                    #         ),
                    #         'robot_ft': tfds.features.Tensor(
                    #             shape=(6,),     # 6
                    #             dtype=np.float32,
                    #         ),
                    #     }),
                    #     'force_torque': tfds.features.FeaturesDict({
                    #         'zeroed': tfds.features.Tensor(
                    #             shape=(6,),     # 6
                    #             dtype=np.float32,
                    #         ),
                    #         'raw': tfds.features.Tensor(
                    #             shape=(6,),     # 6
                    #             dtype=np.float32,
                    #         ),
                    #     }),
                    #     'force_torque_base': tfds.features.FeaturesDict({
                    #         'zeroed': tfds.features.Tensor(
                    #             shape=(6,),     # 6
                    #             dtype=np.float32,
                    #         ),
                    #         'raw': tfds.features.Tensor(
                    #             shape=(6,),     # 6
                    #             dtype=np.float32,
                    #         ),
                    #     }),
                    #     'gripper': tfds.features.FeaturesDict({
                    #         'gripper_command': tfds.features.Sequence(
                    #             tfds.features.Tensor(shape=(), dtype=np.float32)  # Mỗi phần tử là một số
                    #         ),
                    #         'gripper_info': tfds.features.Sequence(
                    #             tfds.features.Tensor(shape=(), dtype=np.float32)
                    #         ),
                    #     }),
                    #     'high_freq_data': tfds.features.FeaturesDict({
                    #         'zeroed': tfds.features.Tensor(
                    #             shape=(6,),
                    #             dtype=np.float32,
                    #         ),
                    #         'raw': tfds.features.Tensor(
                    #             shape=(6,),
                    #             dtype=np.float32,
                    #         ),
                    #         'tcp': tfds.features.Tensor(
                    #             shape=(7,),
                    #             dtype=np.float32,
                    #         ),
                    #     }),
                    #     'joint': tfds.features.Tensor(
                    #         shape=(6,),
                    #         dtype=np.float32,
                    #     ),
                    # }),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'data_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    # now draft for cfg3 only
    def _split_generators(self, dl_manager):
        cfg_folder = dl_manager.manual_dir
        if not cfg_folder:
            raise ValueError("can not found cam dir")
        scene_folders = glob.glob(os.path.join(cfg_folder, "task_*_user_*_scene_*_cfg_*"))

        split_index = int(len(scene_folders) * train_percent)
        train_cfg = scene_folders[:split_index]
        test_cfg = scene_folders[split_index:]

        return {
            "train": self._generate_examples(train_cfg),
            "test": self._generate_examples(test_cfg),
        }

    def _generate_examples(self, scene_folders):
        example_id = 0
        for scene_folder in scene_folders:
            print(f"PROCESS : {scene_folder}")
            # # check metadata ==> no need now
            # meta_path = os.path.join(cam_folder, "metadata.json")
            # if not os.path.exists(meta_path):
            #     continue
            # with open(meta_path, "r") as f:
            #     scene_meta = json.load(f)

            # get task name and all cam info and robot sensor

            task_name = parse_task_from_scene(os.path.basename(scene_folder))
            eng_text, ch_text = get_language_info(task_name)
            cam_dict = build_all_cameras(scene_folder)
            robot_dict = build_robot_data(scene_folder)

            cam_folders = glob.glob(os.path.join(scene_folder, "cam_*"))
            serial_number_list = {os.path.basename(folder).split('_', 1)[1] for folder in cam_folders if os.path.isdir(folder)}

            # each serial number
            for serial_number in serial_number_list:
                steps = sync_and_create_episode(serial_number, cam_dict, robot_dict, eng_text, ch_text)
                if not steps:
                    continue
                postprocess_action_as_next_state(steps)

                sample = {
                    'steps': steps,
                    'episode_metadata': {
                        'data_path': f"{scene_folder}/cam_{serial_number}"
                    }
                }

                yield example_id, sample
                example_id += 1




