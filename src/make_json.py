import fire
import yaml
import subprocess
from pathlib import Path
from os import PathLike, environ
import time


def main(
    cfg_root: PathLike,
    data: PathLike
):
    if not environ.get("DN_BIN"):
        raise Exception("Please specify DN_BIN env variable that points to compiled darknet")

    darknet_path = environ["DN_BIN"] + "/darknet"
    config = yaml.safe_load(open(Path.cwd() / cfg_root / data))
    eval_dir = (Path(config["checkpoint_root"]) / config["result_json_path"]).parent
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_file_dir = Path(config["data_root"]) / config["test_txt"]

    with open(test_file_dir, "r") as tfd:
        test_data = [line.strip() for line in tfd if line.strip()]

    with open(test_file_dir, "r") as f:
        start_time = time.time()

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
            Path(config["checkpoint_root"]) / config['result_json_path'],
            '-thresh',
            str(config['darknet_thresh'])
        ], stdin=f)

        end_time = time.time()
        elapsed = end_time - start_time

    with open(str(Path(config["checkpoint_root"]) / "timing.log"), "w") as t:
        print(f"Test dataset: {len(test_data)}")
        print(f"Average time: {(elapsed / len(test_data)):.3f} seconds\n")
        print(f"Total time: {elapsed:.3f} seconds")
        t.write(f"Average time: {(elapsed / len(test_data)):.3f} seconds\n")
        t.write(f"Total time: {elapsed:.3f} seconds\n")

if __name__ == "__main__":
    fire.Fire(main)
