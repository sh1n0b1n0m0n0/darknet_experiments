# /home/alexsh/roadview_research/code/darknet/darknet detector test cfg/obj.data cfg/yolo.cfg runs/train/backup/yolo_best.weights -ext_output -dont_show -out result.json < data/prepared/valid.txt -thresh 0.5
import yaml
from pathlib import Path
import subprocess


def main():
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["evaluate"]
    with open(params['valid_txt'], 'r') as f:
        p = subprocess.run(["/home/alexsh/roadview_research/code/darknet/darknet", 
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
                        str(params['thresh'])], stdin=f)
        print(p)

if __name__ == "__main__":
    main()