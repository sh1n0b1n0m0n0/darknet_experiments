import fire
import yaml
from pathlib import Path
from os import PathLike, chdir
import subprocess


def main(
    data: PathLike
):
    ODM_path = "./Object-Detection-Metrics/"
    pascalvoc_path = "pascalvoc.py"
    config = yaml.safe_load(open(Path(data)))
    cwd = Path().resolve()

    (cwd / config["odm_result_path"]).mkdir(parents=True, exist_ok=True)

    chdir(ODM_path)
    subprocess.run([
        "python",
        pascalvoc_path,
        "-img",
        str(cwd / config["test_txt"]),
        "-gt",
        str(cwd / config["gt_path"]),
        "-det",
        str(cwd / config["det_path"]),
        "-sp",
        str(cwd / config["odm_result_path"]),
        "--noplot"
    ])

if __name__ == "__main__":
    fire.Fire(main)
