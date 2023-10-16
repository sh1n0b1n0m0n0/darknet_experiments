import yaml
from pathlib import Path
import subprocess
import os


def main():
    ODM_path = "src/Object-Detection-Metrics/"
    pascalvoc_path = "pascalvoc.py"
    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["evaluate"]
    cwd = Path().resolve()

    (cwd / paths["save_path"]).mkdir(parents=True, exist_ok=True)

    os.chdir(ODM_path)
    subprocess.run([
        "python",
        pascalvoc_path,
        "-gt",
        str(cwd / paths["gt_path"]),
        "-det",
        str(cwd / paths["det_path"]),
        "-sp",
        str(cwd / paths["save_path"]),
        "--noplot"
    ])

if __name__ == "__main__":
    main()
