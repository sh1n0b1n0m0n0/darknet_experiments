import yaml
from pathlib import Path
from tqdm import tqdm
from inexlib.object_detection.darknet import to_darknet_config_file, from_darknet_config_file


def change_darknet_config_params(cfg_path: Path, params_names, params):

    def find_index(cfg_lst, parameter):
        return [cfg_lst[0].index(i) for i in cfg_lst[0] if (str(parameter) in i) and ("#" not in i)][0]

    cfg_list = from_darknet_config_file(cfg_path)
    for name in tqdm(params_names):
        idx = find_index(cfg_list, name)
        cfg_list[0][idx] = f"{name}={params[name]}"

    return cfg_list


def main():
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["prepare"]
    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["train"]
    cfg_file_path = Path(paths["cfg"])
    assert cfg_file_path.is_file()

    params_names = ["batch",
                    "subdivisions",
                    "width",
                    "height",
                    "channels",
                    "momentum",
                    "decay",
                    "learning_rate",
                    "max_batches",
                    "burn_in",
                    "policy"]

    print(f"Rewriting {cfg_file_path}...")
    cfg_list = change_darknet_config_params(cfg_file_path, params_names, params)
    to_darknet_config_file(cfg_list, cfg_file_path)
    print("...done")


if __name__ == '__main__':
    main()
