import pickle
import pprint

rollout_path = "rollout/gns_rpf2d_20251212-021503/best/metrics2025_12_12_05_12_04.pkl"

with open(rollout_path, "rb") as f:
    rollout = pickle.load(f)

print(rollout["rollout_1"])
