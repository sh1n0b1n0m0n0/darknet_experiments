import pandas as pd
from pathlib import Path
from os import PathLike
from typing import List
import yaml
import fire
from sklearn.model_selection import train_test_split
import os


# def make_obj_data_file(data_folder, backup_folder, num_of_classes):
#     '''
#         train = /<data_folder>/train.txt
#         names = /<data_folder>/obj.names
#         backup = /<backup_folder>/backup
#         classes = <num_of_classes>
#         valid = /<data_folder>/valid.txt
#         test = /<data_folder>/valid.txt

#     '''
#     (Path(backup_folder) / "backup").mkdir(parents=True, exist_ok=True)

#     obj_data_path = Path(data_folder) / "obj.data"
#     with open(obj_data_path, "w") as f:
#         # data = f"train = {data_folder}/train.txt\nnames = {data_folder}/obj.names\nbackup = {backup_folder}/backup\nclasses = {num_of_classes}\nvalid = {data_folder}/valid.txt\ntest = {data_folder}/valid.txt"
#         # f.write(data)
#         f.write("train = " + data_folder + "/train.txt\n")
#         f.write("names = " + data_folder + "/obj.names\n")
#         f.write("backup = " + backup_folder + "/backup\n")
#         f.write("classes = " + str(num_of_classes) + "\n")
#         f.write("valid = " + data_folder + "/valid.txt\n")
#         f.write("test = " + data_folder + "/valid.txt\n")


# def make_obj_names_file(data_folder, classes):
#     '''
#         class name
#         class name
#     '''
#     obj_names_path = Path(data_folder) / "obj.names"

#     with open(obj_names_path, "a") as f:
#         for cl in classes:
#             f.write(cl)
#             f.write("\n")


def make_folder(root_path, folder_name):
    # root_path.mkdir(parents=True, exist_ok=True)
    # data_list = [i.stem for i in root_path.iterdir()]
    # path = "{}/{}_{:03d}".format(root_path.as_posix(), folder_name, len(data_list)+1)
    # path = f'{root_path.as_posix()}/{folder_name}_{len(data_list)+1:03d}'
    path = f'{root_path.as_posix()}/{folder_name}'
    # print(f'{folder_name}_{len(data_list)+1:03d} created!')
    Path(path).mkdir(parents=True, exist_ok=True)

    return path


def main():
    root_path = Path(__file__).parent
    params = yaml.safe_load(open(Path.cwd() / Path("params.yaml")))["prepare"]

    datasets = params["datasets"]
    train_size = params["train_size"]
    seed = params["seed"]
    classes = params["classes"]
    datasets_path = 'data'
    data = []
    types = ["*.jpg", "*.jp*g", "*.png"]
    # print(root_path)
    # print(Path.cwd())
    for t in types:
        for proj in datasets:
            try:
                for img in (Path.cwd() / Path(datasets_path) / str(proj)).glob(t): 
                    data.append(str(img))
            except ValueError:
                print("Dataset is Empty!")

    df = pd.DataFrame(data, columns=["imgs"])
    df_train, df_valid = train_test_split(df, train_size=train_size, random_state=seed)
    train_set_list = df_train["imgs"].tolist()
    valid_set_list = df_valid["imgs"].tolist()

    prepared_dir = Path.cwd() / Path(datasets_path) / Path('prepared')
    train_val_path = make_folder(prepared_dir, '')
    train_txt_dir = Path(train_val_path) / "train.txt"
    valid_txt_dir = Path(train_val_path) / "valid.txt"
    
    runs_path = make_folder(Path.cwd() / Path('runs'), 'train')
    eval_path = make_folder(Path.cwd() / Path('runs'), 'eval')
    print(runs_path)
    (Path(runs_path) / "backup").mkdir(parents=True, exist_ok=True)
    # make_obj_data_file(train_val_path, runs_path, 18)
    # make_obj_names_file(train_val_path, classes)

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
