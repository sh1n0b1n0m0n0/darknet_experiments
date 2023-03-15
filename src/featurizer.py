import yaml
from pathlib import Path
from itertools import cycle, islice
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


def change_sgdr_policy_with_steps(max_batches, batch, n_img_in_trainset):
    #pattern = [2.0, 2.0, 0.5, 0.5, 0.1, 2.0, 0.5, 0.5, 10]
    pattern = [1.5, 1.25, 0.75, 0.5, 0.25, 3.0]

    max_epochs = round((max_batches * batch) / n_img_in_trainset) + 1
    iter_in_epoch = n_img_in_trainset / batch

    steps = ",".join(str(round(iter_in_epoch * n)) for n in range(1, max_epochs, 10))
    n_steps = len([int(n) for n in steps.split(",")])
    scales = ",".join(str(i) for i in list(islice(cycle(pattern), n_steps)))
    
    return {"steps": steps, "scales": scales}


def add_to_params_yaml(yaml_path, param_dict):
    with open(yaml_path,'r') as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile)
        cur_yaml["prepare"].update(param_dict)

    with open(yaml_path,'w') as yamlfile:
        yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump


def dataset_length(dataset_path):
    with open(dataset_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["prepare"]
    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["train"]
    datasets_paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["check"]

    trainset_path = Path(datasets_paths["train_txt"])
    validset_path = Path(datasets_paths["valid_txt"])
    cfg_file_path = Path(paths["cfg"])

    assert cfg_file_path.is_file()
    assert trainset_path.is_file()
    assert validset_path.is_file()

    if params["policy"] == "steps":
        trainset_length = dataset_length(trainset_path)
        policy_sgdr = change_sgdr_policy_with_steps(params["max_batches"], params["batch"], trainset_length)
        add_to_params_yaml(Path("params.yaml"), policy_sgdr)
        # re-open params
        params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["prepare"]
    
    params_names = ["batch",
                    "subdivisions",
                    "width",
                    "height",
                    "channels",
                    "momentum",
                    "decay",
                    "learning_rate",
                    "max_batches",
                    "burn_in"]
    
    print(f"Rewriting {cfg_file_path}...")
    cfg_list = change_darknet_config_params(cfg_file_path, params_names, params)
    to_darknet_config_file(cfg_list, cfg_file_path)
    print("...done")


if __name__ == '__main__':
    main()
