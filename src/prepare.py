import pandas as pd
from pathlib import Path
from os import PathLike
from typing import List
import yaml
from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm


def make_obj_data_file(data_folder, backup_folder, num_of_classes):
    '''
        train = /<data_folder>/train.txt
        names = /<data_folder>/obj.names
        backup = /<backup_folder>/backup
        classes = <num_of_classes>
        valid = /<data_folder>/valid.txt
        test = /<data_folder>/valid.txt

    '''
    print(f"Making obj_data_file in {str(data_folder)}")

    obj_data_path = Path(data_folder) / "obj.data"
    with open(obj_data_path, "w") as f:
        f.write("train = " + str(Path(data_folder) / "train.txt") + "\n")
        f.write("names = " + "cfg/obj.names" + "\n")
        f.write("backup = " + backup_folder + "\n")
        f.write("classes = " + str(num_of_classes) + "\n")
        f.write("valid = " + str(Path(data_folder) / "valid.txt") + "\n")
        f.write( "test = " + str(Path(data_folder) / "valid.txt") + "\n")


def make_obj_names_file(data_folder, classes):
    '''
        class name
        class name
    '''
    obj_names_path = Path(data_folder) / "obj.names"

    with open(obj_names_path, "a") as f:
        for cl in classes:
            f.write(cl)
            f.write("\n")


def make_folder(root_path, folder_name):
    path = f'{root_path.as_posix()}/{folder_name}'
    Path(path).mkdir(parents=True, exist_ok=True)

    return path


def main():
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["prepare"]

    datasets = params["datasets"]
    train_size = params["train_size"]
    seed = params["seed"]
    classes = params["names"]

    datasets_path = 'data'
    prepared_path = Path(datasets_path) / Path('prepared')
    prepared_dir = Path.cwd() / prepared_path
    train_val_path = make_folder(prepared_dir, '')

    train_txt_dir = Path(train_val_path) / "train.txt"
    valid_txt_dir = Path(train_val_path) / "valid.txt"

    runs_path = make_folder(Path('runs'), 'train')
    eval_path = make_folder(Path('runs'), 'eval')
    backup_path = make_folder(Path(runs_path), 'backup')

    if params["use_default_train_valid"]:
        default_train = "data/default_prepared/train.txt"
        default_valid = "data/default_prepared/valid.txt"
        df_train_list = []
        df_valid_list = []

        with open(default_train, "r") as f:
            t_lines = f.readlines()

        df_train = pd.DataFrame(t_lines, columns=["imgs"])

        with open(default_valid, "r") as f:
            v_lines = f.readlines()

        df_valid = pd.DataFrame(v_lines, columns=["imgs"])

        for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
            img_path = Path(row["imgs"])
            df_train_list.append(str(img_path))

        for index, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
            img_path = Path(row["imgs"])
            df_valid_list.append(str(img_path))

        with open(str(train_txt_dir), "w") as f:
            for img in df_train_list:
                f.write(str(img))

        with open(str(valid_txt_dir), "w") as f:
            for img in df_valid_list:
                f.write(str(img))

        make_obj_data_file(prepared_path, backup_path, len(classes))

        print(f"train size:{train_size} seed:{seed}\n")
    else:
        data = []
        types = ["*.jpg", "*.jp*g", "*.png"]

        for t in types:
            for proj in tqdm(datasets):
                try:
                    for img in (Path(datasets_path) / str(proj)).glob(t):
                        data.append(str(img))
                except ValueError:
                    print("Dataset is Empty!")

        df = pd.DataFrame(data, columns=["imgs"])
        df_train, df_valid = train_test_split(
            df, train_size=train_size, random_state=seed)
        train_set_list = df_train["imgs"].tolist()
        valid_set_list = df_valid["imgs"].tolist()

        make_obj_data_file(prepared_path, backup_path, len(classes))

        with open(str(train_txt_dir), "w") as f:
            for img in train_set_list:
                f.write(str(img))
                f.write("\n")

        with open(str(valid_txt_dir), "w") as f:
            for img in valid_set_list:
                f.write(str(img))
                f.write("\n")

        print(f"train size:{train_size} seed:{seed}\n")


if __name__ == "__main__":
    main()
