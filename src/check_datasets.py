from pathlib import Path, PosixPath
import yaml


def check(path: Path):
    with open(path, "r") as f:
        for line in f:
            assert Path(line.strip()).is_file(), f"doesn't exist {line.strip()}"
        print(f'{path} is Checked')


def main():
    paths = yaml.safe_load(open(Path.cwd() / Path("paths.yaml")))["check"]
    train_txt = Path(paths["train_txt"])
    valid_txt = Path(paths["valid_txt"])

    check(train_txt)
    check(valid_txt)

    
    
if __name__ == "__main__":
    main()
