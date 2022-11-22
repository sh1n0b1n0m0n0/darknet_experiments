# cd /home/alexsh/roadview_research/code/darknet
# ./darknet detector train -dont_show -map cfg/obj.data /cfg/yolo.cfg
# /home/alexsh/roadview_research/code/darknet/darknet detector train -dont_show -map cfg/obj.data cfg/yolo.cfg
import yaml
from pathlib import Path
import subprocess
import os
import shutil

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


    existing_weights = list(filter(
        lambda x: x.stem == "yolo_best",
        list(Path("runs/train/backup/").glob("*.weights")) + list(Path().glob("*.weights"))
    ))

    if len(existing_weights):
        best_weights_file_path = existing_weights[0]
        subprocess_params.append(str(best_weights_file_path))
        weights_path = Path("runs/train/backup/")
        weights_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(best_weights_file_path), str(weights_path / best_weights_file_path.name))


    print(f"train.py: running {' '.join(subprocess_params)}")
    subprocess.run(subprocess_params)
    shutil.copyfile("chart.png", str(weights_path / "chart.png"))
    shutil.copyfile("runs/train/backup/yolo_best.weights", "yolo_best.weights")

if __name__ == "__main__":
    main()
