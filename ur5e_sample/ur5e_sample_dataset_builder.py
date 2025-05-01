from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.append('.')
from pathlib import Path
from ur5e_sample.conversion_utils import MultiThreadedDatasetBuilder


def load_raw_images_per_camera_from_mp4(ep_path: Path, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        video_path = ep_path / f"{camera}_image.rmb.mp4"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, dsize=(256, 256))
        array = np.reshape(resized_frame, (1, 256, 256, 3))
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            resized_frame = cv2.resize(frame, dsize=(256, 256))
            resized_frame = np.reshape(resized_frame, (1, 256, 256, 3))       
            array = np.append(array, resized_frame, axis=0)
        cap.release()

        imgs_per_cam[camera] = array
    return imgs_per_cam


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path):
        # Load raw data
        action = "command_joint_pos"
        state = "measured_joint_pos"
        cameras = ["front_rgb", "hand_rgb"]
        with h5py.File(f"{episode_path}/main.rmb.hdf5", "r") as F:
            actions = F[f"/{action}"][()]
            states = F[f"/{state}"][()]
        imgs_per_cam = load_raw_images_per_camera_from_mp4(
            Path(episode_path),
            cameras,
        )

        # Get language instruction
        # Assumes filepaths look like: "/PATH/TO/ALOHA/PREPROCESSED/DATASETS/<dataset_name>/train/episode_0.hdf5"
        raw_file_string = episode_path.split('/')[-3]  # E.g., '/scr/moojink/data/aloha1_preprocessed/put_green_pepper_into_pot/train/episode_0.hdf5' -> put_green_pepper_into_pot
        #command = " ".join(raw_file_string.split("_"))
        
        command = raw_file_string

        # Assemble episode: here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    f'{cameras[0]}_image': imgs_per_cam[cameras[0]][i],
                    f'{cameras[1]}_image': imgs_per_cam[cameras[1]][i],
                    'state': np.asarray(states[i], np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        # Create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # If you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class Ur5eSample(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 1              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1    # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'front_rgb_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'hand_rgb_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Hand camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot arm action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
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
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/home/y_kitagawa/デスクトップ/real/RoboManipBaselines/robo_manip_baselines/dataset/preprocessed/RealUR5ePutMonotoneBoxIntoBox_20250421/train/*.rmb"),
            "val": glob.glob("/home/y_kitagawa/デスクトップ/real/RoboManipBaselines/robo_manip_baselines/dataset/preprocessed/RealUR5ePutMonotoneBoxIntoBox_20250421/val/*.rmb"),
        }
