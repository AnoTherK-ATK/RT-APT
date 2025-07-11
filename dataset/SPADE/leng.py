import json
files = ["files_cleaned.json", "idle_cleaned.json", "reverse_cleaned.json", "services_cleaned.json", "wget_cleaned.json", "youtube_cleaned.json"]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(file, len(data))