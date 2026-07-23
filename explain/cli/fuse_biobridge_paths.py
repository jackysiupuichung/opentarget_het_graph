"""Fuse per-seed PaGE-Link path explanations into the case-study JSONs.

Reads the per-seed `per_pair_paths.parquet` + `per_pair_edges.parquet` written
by pagelink_explain.py for each grouped-ensemble seed, pools paths across seeds,
and emits the two JSONs the case-study plotting/temporal scripts consume:

  biobridge_paths_totalcost.json : {"<ENSG>|<EFO>": [ {n, tc, E:[[et, "t#i","t#j", m_e], ...]}, ... ]}
  biobridge_names.json           : {"genes": {ENSG:name}, "dis": {EFO:name}}

Fusion: a path is keyed by its ordered edge-type + node-token chain; the same
chain recovered under multiple seeds is pooled, its per-edge mask weight (m_e)
and total_cost taken as the MEDIAN across the seeds that found it. Median (not
mean/max) mirrors the deployed prediction, which is a rank-average across the 5
seeds -- both are outlier-robust central estimators that suppress a single seed
with an inflated mask scale. Paths are ranked by median total_cost (PaGE-Link
concise-path scoring). Only the case-study pairs in CASES are kept.

Paths already respect the receptive-field cap (<= max_hops) because the cap is
enforced upstream in enforce_paths(); this script does not re-cap, but asserts it.

Run on any node (parquet read only):
  .venv/bin/python fuse_biobridge_paths.py
"""
import glob, json, os, re
import statistics
import pandas as pd


def _median(xs):
    return statistics.median(xs)

SEED_ROOT = ("headline_results/evaluate_advancement/pagelink_biobridge")
OUT_PATHS = "/gpfs/scratch/bty414/biobridge_paths_totalcost.json"
OUT_NAMES = "/gpfs/scratch/bty414/biobridge_names.json"
MAX_HOPS = 4     # receptive field of the 2-layer model (2 hops per endpoint)
TOP_K = 6        # paths kept per pair

# case-study pairs to retain (target ENSG, disease EFO)
CASES = {
    "ENSG00000120217|EFO_0000588",   # CD274 (PD-L1) -> mesothelioma
    "ENSG00000112116|EFO_0003778",   # IL17F -> psoriatic arthritis
    "ENSG00000181847|EFO_0003060",   # TIGIT -> non-small cell lung carcinoma
}

# a parquet `path` node token is "type#idx"; an edge segment is
# "type#idx -> [type::rel::type] type#idx -> ...". Parse into node tokens + ets.
_NODE = re.compile(r"([a-z_]+#\d+)")
_ET = re.compile(r"\[([^\]]+)\]")


def parse_path(path_str):
    """Return (node_tokens, edge_types) for a parquet path string."""
    nodes = _NODE.findall(path_str)
    ets = _ET.findall(path_str)
    return nodes, ets


def parse_named(named_str):
    """Map "type#idx (ACC (name))"... segments to {token: (acc, name)}.
    The named path is "ACC (NAME) -> [et] ACC (NAME) -> ..."; we align it to the
    token order of the plain path (same segment count)."""
    # accession is the leading token of each node segment; capture "ACC (NAME)".
    segs = re.split(r"\s*->\s*\[[^\]]+\]\s*", named_str)
    out = []
    for s in segs:
        m = re.match(r"\s*(\S+)\s*(?:\(([^)]*)\))?", s.strip())
        if m:
            acc, name = m.group(1), (m.group(2) or m.group(1))
            out.append((acc, name))
    return out


def main():
    seed_dirs = sorted(glob.glob(f"{SEED_ROOT}/s*"))
    assert seed_dirs, f"no seed dirs under {SEED_ROOT}"
    print(f"fusing {len(seed_dirs)} seeds: {[os.path.basename(d) for d in seed_dirs]}")

    genes, dis = {}, {}
    # pooled[key][chain] = {"n":, "ets":, "toks":, "m_by_edge":[list of lists], "tc":[list]}
    pooled = {}

    for sd in seed_dirs:
        paths = pd.read_parquet(f"{sd}/per_pair_paths.parquet")
        edges = pd.read_parquet(f"{sd}/per_pair_edges.parquet")
        # per-(pair,edge) mask weight for this seed
        ekey = {(r.target_id, r.disease_id, r.edge_type, int(r.src), int(r.dst)): float(r.mask_weight)
                for r in edges.itertuples()}
        for r in paths.itertuples():
            key = f"{r.target_id}|{r.disease_id}"
            if key not in CASES:
                continue
            if int(r.n_hops) > MAX_HOPS:
                continue
            toks, ets = parse_path(r.path)
            if len(ets) != len(toks) - 1:
                continue
            # collect names from the named path
            named = parse_named(r.path_named)
            for tok, (acc, name) in zip(toks, named):
                nt = tok.split("#")[0]
                (genes if nt == "target" else dis if nt == "disease" else {}) \
                    .__setitem__(acc, name) if nt in ("target", "disease") else None
            chain = tuple(zip(ets, toks[:-1], toks[1:]))
            slot = pooled.setdefault(key, {}).setdefault(
                chain, {"n": len(ets), "ets": ets, "toks": toks,
                        "m_by_edge": [[] for _ in ets], "tc": []})
            slot["tc"].append(float(r.total_cost))
            for i, (et, a, b) in enumerate(zip(ets, toks[:-1], toks[1:])):
                _, ai = a.split("#"); _, bi = b.split("#")
                me = ekey.get((r.target_id, r.disease_id, et, int(ai), int(bi)))
                if me is not None:
                    slot["m_by_edge"][i].append(me)

    # emit: per key, rank pooled chains by MEDIAN total_cost, top-K
    out = {}
    for key, chains in pooled.items():
        ranked = sorted(chains.values(), key=lambda s: _median(s["tc"]))
        objs = []
        for s in ranked[:TOP_K]:
            E = []
            for et, a, b, mlist in zip(s["ets"], s["toks"][:-1], s["toks"][1:], s["m_by_edge"]):
                m = round(_median(mlist), 3) if mlist else None
                E.append([et, a, b, m])
            objs.append({"n": s["n"], "tc": round(_median(s["tc"]), 3), "E": E})
        out[key] = objs
        assert all(o["n"] <= MAX_HOPS for o in objs), f"{key} has a >{MAX_HOPS}-hop path"

    with open(OUT_PATHS, "w") as fh:
        json.dump(out, fh, indent=1)
    with open(OUT_NAMES, "w") as fh:
        json.dump({"genes": genes, "dis": dis}, fh, indent=1)
    print(f"wrote {OUT_PATHS}: {[(k, len(v)) for k, v in out.items()]}")
    print(f"wrote {OUT_NAMES}: {len(genes)} genes, {len(dis)} diseases")
    for k, v in out.items():
        print(f"  {k}: hops {[o['n'] for o in v]}")


if __name__ == "__main__":
    main()
