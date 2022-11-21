# cd /home/alexsh/roadview_research/code/darknet
# ./darknet detector train -dont_show -map cfg/obj.data /cfg/yolo.cfg
# /home/alexsh/roadview_research/code/darknet/darknet detector train -dont_show -map cfg/obj.data cfg/yolo.cfg
import yaml
from pathlib import Path
import subprocess
import os

def main():
    if not os.environ.get("DN_BIN"):
        raise Exception("please specify DN_BIN env variable that points to compiled darknet")
    darknet_path = os.environ["DN_BIN"] + "/darknet"

    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["train"]

    subprocess_params = [darknet_path,
        "detector",
        "train",
        "-dont_show",
        "-map",
        paths['obj_data'],
        paths['cfg']
    ]

    existing_weights = list(Path("runs/train/backup/").glob("*.weights"))

    if len(existing_weights):
        last_weights = None
        if "runs/train/backup/yolo_best.weights" in existing_weights:
            last_weights = "runs/train/backup/yolo_best.weights"
        elif "runs/train/backup/yolo_last.weights" in existing_weights:
            last_weights = "runs/train/backup/yolo_last.weights"
        else:
            last_weights = sorted(list(map(lambda x: str(x), existing_weights)))[-1]
        subprocess_params.append(last_weights)

    print(f"train.py: running {' '.join(subprocess_params)}")
    subprocess.run(subprocess_params)

if __name__ == "__main__":
    main()
