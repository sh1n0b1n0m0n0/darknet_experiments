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

    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["train"]
    subprocess_params = [darknet_path,
                    "detector",
                    "train",
                    "-dont_show",
                    "-map",
                    params['obj_data'],
                    params['cfg']]
    print(f"train.py: running {' '.join(subprocess_params)}")
    subprocess.run(subprocess_params)

if __name__ == "__main__":
    main()
