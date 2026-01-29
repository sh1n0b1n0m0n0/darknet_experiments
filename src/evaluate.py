import fire
import yaml
from pathlib import Path
from os import PathLike, chdir
import subprocess


def main(
    cfg_root: PathLike,
    data: PathLike
):
    ODM_path = "./Object-Detection-Metrics/"
    pascalvoc_path = "pascalvoc.py"
    config_path = Path.cwd().parent / cfg_root / data
    print(config_path)
    config = yaml.safe_load(open(config_path))

    (Path(config["checkpoint_root"]) / config["odm_result_path"]).mkdir(parents=True, exist_ok=True)

    chdir(ODM_path)
    subprocess.run([
        "python",
        pascalvoc_path,
        "-img",
        Path(config["data_root"]) / config["test_txt"], 
        "-gt",
        Path(config["checkpoint_root"]) / config["gt_path"],
        "-det",
        Path(config["checkpoint_root"]) / config["det_path"],
        "-sp",
        Path(config["checkpoint_root"]) / config["odm_result_path"],
        "--noplot"
    ])

if __name__ == "__main__":
    fire.Fire(main)
