import json
import sys

files = ["files_small.json", "idle_small.json", "services_small.json", "wget_small.json", "youtube_small.json"]

data = []
_data = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        _data = json.load(f)
        data.extend(_data)

with open("benign_small.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)