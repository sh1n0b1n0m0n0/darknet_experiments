# darknet_experiments
Pipeline for Darknet
### Dependencies:
```bash
git submodule update --init --recursive
```
export PATH=./venv/bin:$PATH
export DN_BIN= "your PATH to darknet directory"
### Commands:
```bash
dvc repro
```
```bash
dvc repro --no-run-cache
```
