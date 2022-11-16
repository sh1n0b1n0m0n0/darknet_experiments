# cd /home/alexsh/roadview_research/code/darknet
# ./darknet detector train -dont_show -map cfg/obj.data /cfg/yolo.cfg
# /home/alexsh/roadview_research/code/darknet/darknet detector train -dont_show -map cfg/obj.data cfg/yolo.cfg
import yaml
from pathlib import Path
import subprocess


def main():
    darknet_path = str(Path.home()) + "/roadview_research/code/darknet/darknet"
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["train"]
    subprocess.run([darknet_path, 
                    "detector", 
                    "train", 
                    "-dont_show",
                    "-map", 
                    params['obj_data'], 
                    params['cfg']])

if __name__ == "__main__":
    main()
