# RLDS Dataset Conversion


## Installation

```
conda env create -f environment_ubuntu.yml
```
```
conda activate rlds_env
```

## Converting your Own Dataset to RLDS

cd my_datasets/rh20t

tfds build --data_dir=/home/nhattx/Workspace/VR/Study_robotics/source/rh20t_data_builder/rh20t_dataset/data_output --manual_dir=/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_unrar/RH20T_cfg3


