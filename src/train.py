import yaml
import fire
from pathlib import Path
import subprocess
from os import PathLike, environ
import shutil

environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used


def main(
    cfg_root: PathLike,
    data: PathLike,
    checkpoint: PathLike ="yolo_best.weights"
):
    if not environ.get("DN_BIN"):
        raise Exception("Please specify DN_BIN env variable that points to compiled darknet")
    darknet_path = environ["DN_BIN"] + "/darknet"

    config = yaml.safe_load(open(Path.cwd() / cfg_root / data))

    checkpoint_path = Path("runs/train/backup/")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    subprocess_params = [darknet_path,
        "detector",
        "train",
        "-dont_show",
        "-map",
        config['data_file'],
        config['cfg_file']
    ]

    existing_weights = Path(checkpoint)

    if existing_weights.is_file():
        print("Re-training...")
        subprocess_params.append(str(existing_weights))
        shutil.copyfile(existing_weights, checkpoint_path / existing_weights.name)

    print(f"train.py: running {' '.join(subprocess_params)}")
    subprocess.run(subprocess_params)

    if Path("chart.png").is_file():
        shutil.copyfile("chart.png", str(checkpoint_path / "chart.png"))


if __name__ == "__main__":
   fire.Fire(main)
