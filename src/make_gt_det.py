import yaml
from pathlib import Path
import subprocess


def main():
    darknet_helper_path = "./src/darknet_train_helper/"
    gt_path = darknet_helper_path + "convert_darknet_gt_to_pascal_voc_gt.py"
    det_path = darknet_helper_path + "convert_darknet_json_detect_to_pascal_voc_detect.py"

    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["make_gt_det"]
    subprocess.run(["python",
                    gt_path,
                    "-file_path", 
                    params["valid_txt"], 
                    "-names",
                    params["obj_names"], 
                    "-save_path",
                    params["gt_path"]])

    subprocess.run(["python",
                    det_path,
                    "-file_path", 
                    params["result"], 
                    "-save_path",
                    params["det_path"]])

if __name__ == "__main__":
    main()
