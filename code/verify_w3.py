"""verify_w3.py -- S3 completion check: shard count, sample count, SHA256, class/branch balance."""
import json, os, glob, hashlib
import numpy as np

ROOT = r"F:\paper\2\P2\data"
print(f"{'split':6s}{'shards':>7s}{'samples':>9s}{'GB':>7s}{'sha_ok':>8s}"
      f"{'P':>6s}{'U':>6s}{'R':>6s}  branches")
tot_n = 0; tot_b = 0; ok_all = True
for sp in ("train", "val", "test"):
    d = os.path.join(ROOT, sp)
    man = json.load(open(os.path.join(d, "manifest_000.json")))
    # ONLY manifest-listed shards: a stale *.npz left in the folder used to pollute the
    # class counts (2026-11-17: 4 pilot shards added 88 phantom samples to train).
    files = [os.path.join(d, s["path"]) for s in man["shards"]]
    stale = sorted(set(glob.glob(os.path.join(d, "*.npz"))) - set(files))
    if stale:
        print(f"  !! {sp}: {len(stale)} npz not in manifest (ignored): "
              f"{[os.path.basename(p) for p in stale]}")
    ok = True
    for s in man["shards"]:
        p = os.path.join(d, s["path"])
        if not os.path.exists(p):
            ok = False; continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != s["sha256"]:
            ok = False
    n = sum(s["n"] for s in man["shards"])
    gb = sum(os.path.getsize(p) for p in files) / 1e9
    # class / branch balance from meta of all shards
    from collections import Counter
    cl, br = Counter(), Counter()
    for p in files:
        z = np.load(p)
        for s in z["meta"]:
            m = json.loads(s)
            cl[m["cls"]] += 1; br[m["branch"]] += 1
    tot_n += n; tot_b += gb; ok_all &= ok
    print(f"{sp:6s}{len(man['shards']):7d}{n:9d}{gb:7.2f}{str(ok):>8s}"
          f"{cl.get('P',0):6d}{cl.get('U',0):6d}{cl.get('R',0):6d}  {dict(br)}")
print(f"\nTOTAL samples = {tot_n}   size = {tot_b:.2f} GB   all SHA256 ok = {ok_all}")
for sp in ("train", "val", "test"):
    man = json.load(open(os.path.join(ROOT, sp, "manifest_000.json")))
    print(f"  {sp:5s} blocks={man['blocks']} buffer={man['buffer']} "
          f"dtd_cats={len(man.get('dtd_categories', []))}")
