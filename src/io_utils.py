import os
import json


def load_txt(path, sep=None):
    print(f"Read text file from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if sep:
        print(f"Split each line with sep {sep}.")
        return [line.strip().split(sep) for line in lines]
    else:
        return [line.strip() for line in lines]

def save_txt(data, path, sep=" "):
    item_len = len(data[0])
    if item_len > 1 and sep is not None:
        data = [sep.join(d) for d in data]
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.writelines(line+"\n")
    print(f"Data has been saved to {path}.")

def load_json(path):
    print(f"Read text file from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        lines = json.load(f)
    return lines

def load_jsonl(path):
    print(f"Read text file from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.writelines(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Data has been saved to {path}.")

def check_dirs(path):
    if not os.path.exists(path):
        print(f"{path} does not exist, creat it!")
        os.makedirs(path)
    return True
