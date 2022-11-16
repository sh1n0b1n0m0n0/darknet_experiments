import yaml
from pathlib import Path
import subprocess
import os


def main():
    ODM_path = str(Path.home()) + "/Object-Detection-Metrics/"
    pascalvoc_path = "pascalvoc.py"
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["evaluate"]
    Path(str(Path.home()) + "/darknet_experiments/" + params["save_path"]).mkdir(parents=True, exist_ok=True)
    os.chdir(ODM_path)
    subprocess.run(["python",
                    pascalvoc_path,
                    "-gt", 
                    str(Path.home()) + "/darknet_experiments/" + params["gt_path"], 
                    "-det", 
                    str(Path.home()) + "/darknet_experiments/" + params["det_path"], 
                    "-sp", 
                    str(Path.home()) + "/darknet_experiments/" + params["save_path"]])

if __name__ == "__main__":
    main()
