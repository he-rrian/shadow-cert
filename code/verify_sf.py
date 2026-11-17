"""verify_sf.py -- SynthFringe export check: manifest, SHA256, unique-scene rule, schema.

Run after w3_export_sf.py (locally or on the training server):
  python code/verify_sf.py
"""
import json, os, glob, hashlib
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VINF = ["sliver", "arc", "blob"]
OTHER = ["edge", "cut", "degraded"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def check(sp):
    d = os.path.join(ROOT, sp)
    man = json.load(open(os.path.join(d, "manifest.json")))
    files = [os.path.join(d, s["path"]) for s in man["shards"]]
    stale = sorted(set(glob.glob(os.path.join(d, "*.npz"))) - set(files))
    sha_ok = True
    for s in man["shards"]:
        p = os.path.join(d, s["path"])
        if not os.path.exists(p) or sha(p) != s["sha256"]:
            sha_ok = False
    n_man = sum(s["n"] for s in man["shards"])
    metas, dtypes = [], set()
    for p in files:
        z = np.load(p)
        dtypes |= {(k, str(z[k].dtype)) for k in ("I", "phi", "roi", "dead", "orr", "kmap")}
        metas += [json.loads(m) for m in z["meta"]]
    per = Counter(m["branch"] for m in metas)
    uniq_all = len({m["sid"] for m in metas})
    vin = [m for m in metas if m.get("vinf") is not None]
    uniq_vin = len({m["sid"] for m in vin})
    blocks = Counter(m.get("block", int(m["sid"]) // 200) for m in vin)
    ok = (sha_ok and n_man == len(metas) == man["n_samples"] and not stale)
    ok &= man.get("n_vinf") == len(vin) == 450 and uniq_vin == 450
    ok &= all(per[b] == 150 for b in VINF) and all(per[b] == 80 for b in OTHER)
    print(f"{sp:8s} n={len(metas):4d} vinf={len(vin):3d} uniq_vin={uniq_vin:3d} "
          f"uniq_all={uniq_all:3d} sha={'OK' if sha_ok else 'FAIL'} "
          f"per={dict(per)} blocks={dict(blocks)}")
    if stale:
        print(f"  !! stale npz not in manifest: {[os.path.basename(p) for p in stale]}")
    if not ok:
        print("  !! CHECK FAILED")
    return ok


if __name__ == "__main__":
    allok = all(check(sp) for sp in ("sf_val", "sf_test"))
    print("\nALL OK" if allok else "\nFAILED")
