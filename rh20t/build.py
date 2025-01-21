import os
from rh20t.rh20t_dataset_builder import Rh20tDataset
import tensorflow_datasets as tfds
import glob
from rh20t.config import root_dir, output_dir

if __name__ == '__main__':
    cfg_folders = glob.glob(os.path.join(root_dir, "RH20T_*"))
    for cfg_folder in cfg_folders:
        # create folder for output
        cfg_code = os.path.basename(cfg_folder)
        cfg_output_path = os.path.join(output_dir, cfg_code)
        os.makedirs(cfg_output_path, exist_ok=True)

        builder = Rh20tDataset(data_dir=cfg_output_path)
        builder.download_and_prepare(download_config=tfds.download.DownloadConfig(
                manual_dir=cfg_folder))
        print(f"{cfg_code} dataset created successfully!")
