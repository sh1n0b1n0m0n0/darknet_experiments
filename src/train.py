# cd /home/alexsh/roadview_research/code/darknet
# ./darknet detector train -dont_show -map cfg/obj.data /cfg/yolo.cfg
# /home/alexsh/roadview_research/code/darknet/darknet detector train -dont_show -map cfg/obj.data cfg/yolo.cfg
import yaml
from pathlib import Path
import subprocess
import os
import shutil

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used


def main():
    if not os.environ.get("DN_BIN"):
        raise Exception("please specify DN_BIN env variable that points to compiled darknet")
    darknet_path = os.environ["DN_BIN"]

    paths = yaml.safe_load(open(Path.cwd() / Path("paths_lpr.yaml")))["train"]

    weights_path = Path("runs/train/backup/")
    weights_path.mkdir(parents=True, exist_ok=True)

    subprocess_params = [darknet_path + "/darknet",
        "detector",
        "train",
        "-dont_show",
        "-map",
        paths['obj_data'],
        paths['cfg']
        
    ]

    existing_weights = Path("yolo_best.weights")

    if existing_weights.is_file():
        print("Re-training...")
        subprocess_params.append(str(existing_weights))
        shutil.copyfile(existing_weights, weights_path / existing_weights)

    print(f"train.py: running {' '.join(subprocess_params)}")
    subprocess.run(subprocess_params)

    if Path("chart.png").is_file():
        shutil.copyfile("chart.png", str(weights_path / "chart.png"))


if __name__ == "__main__":
    main()
