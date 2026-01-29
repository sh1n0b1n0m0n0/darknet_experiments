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
):
    if not environ.get("DN_BIN"):
        raise Exception("Please specify DN_BIN env variable that points to compiled darknet")
    darknet_path = environ["DN_BIN"] + "/darknet"

    config = yaml.safe_load(open(Path.cwd() / cfg_root / data))

    subprocess_params = [darknet_path,
        "detector",
        "map",
        config["data_file"],
        config["cfg_file"],
        config["weights"]
    ]


    print(f"check_mAP.py: running {' '.join(subprocess_params)}")
    subprocess.run(subprocess_params)


if __name__ == "__main__":
   fire.Fire(main)
