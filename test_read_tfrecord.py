import tensorflow as tf
import os

# List the TFRecord files (replace with your actual file paths)
tfrecord_files = [
    "berkeley_cable_routing-test.tfrecord-00000-of-00004",
    "berkeley_cable_routing-test.tfrecord-00001-of-00004",
    "berkeley_cable_routing-test.tfrecord-00002-of-00004",
    "berkeley_cable_routing-test.tfrecord-00003-of-00004",
]
tfrecord_files = [
    os.path.join(
        # "/home/nhattx/Workspace/VR/Study_robotics/dataset/RH20T_rlds/RH20T_cfg3/rh20t_dataset/1.0.0",
        "/home/nhattx/Workspace/VR/Study_robotics/dataset/berkeley_cable_routing/0.1.0",
        f
    ) for f in tfrecord_files
    ]


# Function to parse and decode the raw record as an Example
def parse_raw_example(raw_record):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())  # Convert byte string to Example object
    return example


if __name__ == '__main__':
    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    # Inspect and parse the first 5 raw records
    for raw_record in dataset.take(1):
        example = parse_raw_example(raw_record)
        print(example)  # Print the decoded Example
