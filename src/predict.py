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

    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["predict"]
    detect_path = Path("runs/detect/")
    detect_path.mkdir(parents=True, exist_ok=True)
    test_txt = Path(paths['test_txt'])
    assert test_txt.is_file()

    with open(str(test_txt), "r") as f:
        imgs = f.read().splitlines()
    
    for img in imgs:
        subprocess_params = [darknet_path + "/darknet",
            "detector",
            "test",
            paths['obj_data'],
            paths['cfg'],
            paths['weights'],
            "-thresh",
            "0.25",
            img
        ]

        print(f"predict.py: running {' '.join(subprocess_params)}")
        subprocess.run(subprocess_params)
        shutil.copyfile(paths["pred_image"], str(detect_path / Path(img).name))


if __name__ == "__main__":
    main()
