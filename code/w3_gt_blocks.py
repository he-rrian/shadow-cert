"""w3_gt_blocks.py -- batch cascade GT for SynthFringe blocks 12..17.

Prerequisite for `w3_export_sf.py` (SF-val = blocks 12..14, SF-test = 15..17).
Each scene is unwrapped by `w2_d1_modroi.run()` and written to
`probes/results/gt/{sid}.npz` with keys phi64 / m_gt / amp64.

Safe to kill and restart: existing GT files are skipped.  A 15-scene buffer is
dropped at every 200-scene block boundary, matching w3_export_sf.py.

Run:
  python w3_gt_blocks.py                 # 1020 scenes, ~2-3 h
  python w3_gt_blocks.py --limit 2       # smoke test
  python w3_gt_blocks.py --blocks 12 14  # one split only
"""
import os, sys, time, argparse, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLOCK, BUFFER = 200, 15


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = load("w2mod", os.path.join(HERE, "w2_d1_modroi.py"))
GT = os.path.join(M.RES, "gt")


def scene_ids(blocks):
    ids = []
    for b in blocks:
        lo, hi = b * BLOCK, (b + 1) * BLOCK
        ids += list(range(lo + BUFFER, hi - BUFFER))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs=2, default=(12, 17))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(GT, exist_ok=True)
    ids = scene_ids(range(a.blocks[0], a.blocks[1] + 1))
    if a.limit:
        ids = ids[:a.limit]
    todo = [s for s in ids if not os.path.exists(os.path.join(GT, f"{s:05d}.npz"))]
    print(f"scenes: {len(ids)}  already done: {len(ids)-len(todo)}  to do: {len(todo)}",
          flush=True)
    t0 = time.time()
    ok = bad = 0
    for j, sid in enumerate(todo, 1):
        p = os.path.join(GT, f"{sid:05d}.npz")
        try:
            M.run(f"{sid:05d}", save=True)
            ok += 1
        except Exception as ex:
            bad += 1
            print(f"  {sid:05d}: ERR {type(ex).__name__} {ex}", flush=True)
        if j % 20 == 0 or j == len(todo):
            el = time.time() - t0
            print(f"  {j}/{len(todo)}  ok={ok} bad={bad}  {el:.0f}s  "
                  f"eta={(el/j)*(len(todo)-j):.0f}s", flush=True)
    print(f"done: ok={ok} bad={bad}  gt_dir={GT}", flush=True)


if __name__ == "__main__":
    main()
