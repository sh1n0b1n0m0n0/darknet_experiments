# /home/alexsh/roadview_research/code/darknet/darknet detector test cfg/obj.data cfg/yolo.cfg runs/train/backup/yolo_best.weights -ext_output -dont_show -out result.json < data/prepared/valid.txt -thresh 0.5
import yaml
from pathlib import Path
import subprocess
import os

def main():
    if not os.environ.get("DN_BIN"):
        raise Exception("please specify DN_BIN env variable that points to compiled darknet")
    darknet_path = os.environ["DN_BIN"] + "/darknet"
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["make_json"]
    with open(params['valid_txt'], 'r') as f:
        subprocess.run([
            darknet_path,
            "detector",
            "test",
            '-dont_show',
            params['obj_data'],
            params['cfg'],
            params['weights'],
            '-ext_output',
            '-out',
            params['result'],
            '-thresh',
            str(params['thresh'])
        ], stdin=f)

if __name__ == "__main__":
    main()
