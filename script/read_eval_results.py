import os
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument('--base_dir', type=str, default='', help="truncated output path (before scene)")
args = parser.parse_args()

scenes = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]

mean_d2s = []
mean_s2d = []
overall = []
for scene in scenes:
    data_dir = os.path.join(args.base_dir, f"scan{scene}", "meshes/results.json")
    try:
        with open(data_dir, 'r') as f:
            data = json.load(f)
    except:
        continue
    mean_s2d.append(data["mean_s2d"])
    mean_d2s.append(data["mean_d2s"])
    overall.append(data["overall"])

# print(mean_d2s)
# print(mean_s2d)
print(overall)