# /home/alexsh/roadview_research/code/darknet/darknet detector test cfg/obj.data cfg/yolo.cfg runs/train/backup/yolo_best.weights -ext_output -dont_show -out result.json < data/prepared/valid.txt -thresh 0.5
import fire
import yaml
import subprocess
from pathlib import Path
from os import PathLike, environ


def main(
    data: PathLike
):
    if not environ.get("DN_BIN"):
        raise Exception("please specify DN_BIN env variable that points to compiled darknet")
    darknet_path = environ["DN_BIN"] + "/darknet"
    config = yaml.safe_load(open(Path(data)))
    eval_dir = Path(config['result_json_path']).parent
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config['test_txt'], 'r') as f:
        subprocess.run([
            darknet_path,
            "detector",
            "test",
            '-dont_show',
            config['data_file'],
            config['cfg_file'],
            config['weights'],
            '-ext_output',
            '-out',
            config['result_json_path'],
            '-thresh',
            str(config['darknet_thresh'])
        ], stdin=f)

if __name__ == "__main__":
    fire.Fire(main)
