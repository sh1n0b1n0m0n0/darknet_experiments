import fire
import yaml
from pathlib import Path
from os import PathLike
import subprocess


def main(
    data: PathLike
):
    darknet_helper_path = "./darknet_train_helper/"
    gt_path = darknet_helper_path + "convert_darknet_gt_to_pascal_voc_gt.py"
    det_path = darknet_helper_path + "convert_darknet_json_detect_to_pascal_voc_detect.py"

    config = yaml.safe_load(open(Path(data)))

    # gt script activation
    subprocess.run(["python",
                    gt_path,
                    "-test_txt", 
                    config["test_txt"], 
                    "-names",
                    config["names_file"], 
                    "-save_path",
                    config["gt_path"]])
    # det script activation
    subprocess.run(["python",
                    det_path,
                    "-result_json_path", 
                    config["result_json_path"],
                    "-data",
                    data,
                    "-save_path",
                    config["det_path"]])

    Path(config["odm_result_path"]).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    fire.Fire(main)
