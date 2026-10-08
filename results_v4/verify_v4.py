import sys, json, numpy as np
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import confirmation_v4 as C
P = json.loads((C.ROOT/"PROTOCOL_V4.json").read_text()); models = C.load_models(P["models"]); prior = dict(np.load(C.ROOT/P["arx_prior"]["path"]))
R = json.load(open(C.ROOT/'results_v4'/'confirmation_v4_results.json'))
raw = R["raw"]; seeds = list(range(*P["confirmation"]["plant_seeds_range"])); conds = {c["key"]: c for c in C.conditions(P)}
def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p+"/"+k)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, p+f"[{i}]")
    else: yield p, x
mode = sys.argv[1]
if mode == "analysis":
    A = json.loads(json.dumps(C.analyse(raw, P)))
    d = max(abs(a-b) for (_, a), (_, b) in zip(leaves(A), leaves(R["analysis"])) if isinstance(a, (int, float)) and not isinstance(a, bool))
    print("analysis recompute max |diff|:", d)
else:
    for item in sys.argv[2:]:
        key, i = item.rsplit(":", 1); i = int(i)
        rec = json.loads(json.dumps(C.run_plant(seeds[i], conds[key], P, models, prior)))
        diffs = [abs(a-b) for (_, a), (_, b) in zip(leaves(rec), leaves(raw[key][i])) if isinstance(a, (int, float)) and not isinstance(a, bool)]
        print(key, i, "n leaves", len(diffs), "max |diff|", max(diffs), flush=True)
