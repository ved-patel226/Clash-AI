import json

with open("battle_summary_list.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_json = []

for d in data:
    for v in d["value"]:
        new_json.append(v)

with open("battle_summary_list2.json", "w", encoding="utf-8") as f:
    json.dump(new_json, f)
