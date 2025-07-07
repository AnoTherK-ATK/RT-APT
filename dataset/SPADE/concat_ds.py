import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

with open(sys.argv[2], "r", encoding="utf-8") as f:
    data2 = json.load(f)

data.extend(data2)

with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)