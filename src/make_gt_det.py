import yaml
from pathlib import Path
import subprocess


def main():
    darknet_helper_path = "./src/darknet_train_helper/"
    gt_path = darknet_helper_path + "convert_darknet_gt_to_pascal_voc_gt.py"
    det_path = darknet_helper_path + "convert_darknet_json_detect_to_pascal_voc_detect.py"

    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["make_gt_det"]
    subprocess.run(["python",
                    gt_path,
                    "-file_path", 
                    paths["valid_txt"], 
                    "-names",
                    paths["obj_names"], 
                    "-save_path",
                    paths["gt_path"]])

    subprocess.run(["python",
                    det_path,
                    "-file_path", 
                    paths["result"], 
                    "-save_path",
                    paths["det_path"]])

if __name__ == "__main__":
    main()
