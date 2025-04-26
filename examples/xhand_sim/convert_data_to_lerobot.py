import shutil
import os
import json
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
from pathlib import Path

REPO_NAME = "Chenba/xhand_sim"  # Name of the output dataset, also used for the Hugging Face Hub

def main(data_root: str, *, push_to_hub: bool = False):
    annotation_dir = Path(data_root) / "annotation/train"
    image_root = Path(data_root) / "imgs/train"

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xhand",
        fps=10,
        features={
             "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    # Loop over raw datasets and write episodes to the LeRobot dataset
    # 获取所有 JSON 文件并按名称排序
    all_json_files = sorted(annotation_dir.glob("*.json"))

    # 只取前 50 个编号（从 0 到 49）
    json_files_to_process = [f for f in all_json_files if f.stem.isdigit() and int(f.stem) < 50]

    for json_file in json_files_to_process:
        with open(json_file, "r") as f:
            episode_data = json.load(f)

        episode_id = str(episode_data["episode_id"])
        task_name = episode_data["texts"][0] if episode_data["texts"] else f"task_{episode_id}"
        video_len = episode_data["video_length"]

        states = [s[-12:] for s in episode_data["states"]]
        actions = [a[-12:] for a in episode_data["actions"]]

        for frame_id in range(video_len):
            img_path = image_root / episode_id / "0" / f"{frame_id}.jpg"
            wrist_img_path = image_root / episode_id / "1" / f"{frame_id}.jpg"

            if not img_path.exists() or not wrist_img_path.exists():
                print(f"Missing image or wrist image at frame {frame_id} in episode {episode_id}, skipping.")
                continue

            image = np.array(Image.open(img_path).resize((256, 256)))
            wrist_image = np.array(Image.open(wrist_img_path).resize((256, 256)))

            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": np.array(states[frame_id], dtype=np.float32),
                    "actions": np.array(actions[frame_id], dtype=np.float32),
                }
            )

        dataset.save_episode(task=task_name)

    dataset.consolidate(run_compute_stats=False)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["xhand", "rlds", "custom"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)