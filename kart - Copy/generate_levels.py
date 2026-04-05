import json

file_path = "data/levels.json"

# load file
with open(file_path, "r") as f:
    data = json.load(f)

# add animation
for pose in data:
    ref = pose.get("reference_pose", [])
    pose["animation"] = [ref, ref, ref]

# overwrite file
with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ levels.json updated with animation")