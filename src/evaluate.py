import yaml
from pathlib import Path
import subprocess
import os


def main():
    ODM_path = "src/Object-Detection-Metrics/"
    pascalvoc_path = "pascalvoc.py"
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["evaluate"]
    cwd = Path().resolve()

    (cwd / params["save_path"]).mkdir(parents=True, exist_ok=True)

    os.chdir(ODM_path)
    subprocess.run([
        "python",
        pascalvoc_path,
        "-gt",
        str(cwd / params["gt_path"]),
        "-det",
        str(cwd /  params["det_path"]),
        "-sp",
        str(cwd /  params["save_path"]),
        "--noplot"
    ])

if __name__ == "__main__":
    main()
