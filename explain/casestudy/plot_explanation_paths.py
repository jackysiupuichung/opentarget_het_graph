"""MedKGent-style explanation-path diagrams.

Render a set of explanation paths as node-icon + labelled-edge chains, one row
per path: each node is a coloured circle keyed by entity type (target/gene,
disease, molecule/drug, pathway, GO/phenotype) with the entity name below, and
each edge is an arrow carrying the relation name and optionally the [s,n] edge
score. Mirrors the MedKGent figure style.

Reads the fused-path + name JSON dumped by the case-study extraction, so it is
reproducible. Run on a compute node (needs the graph mappings for score lookup).
"""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

OUT = "/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph/figures/results"
os.makedirs(OUT, exist_ok=True)

# entity-type visual style: (fill, edge, single-letter glyph)
TYPE_STYLE = {
    "target":   ("#d7f0d7", "#2ca02c", "gene"),      # gene/target — green
    "disease":  ("#fde0dc", "#d62728", "dis"),       # disease — red
    "molecule": ("#fff3c4", "#e6b800", "drug"),      # drug — yellow
    "reactome": ("#dbe9fb", "#1f77b4", "path"),      # pathway — blue
    "go":       ("#efe0f7", "#7b4fb0", "GO"),        # GO function — purple
}

REL_ABBR = {
    "clinical_trial_positive": "trial (positive)",
    "clinical_trial_ongoing": "in trial for",
    "clinical_trial_Unknown/Operational": "in trial for",
    "clinical_trial_unmet_efficacy": "trial (unmet)",
    "clinical_trial_adverse_effects": "trial (adverse)",
    "modulated_by": "modulates", "genetic_association": "associates",
    "interacts_with": "interacts", "involved_in": "in pathway",
    "affected_pathway": "affects pathway", "has_function_in": "has function",
    "is_subtype_of": "subtype of", "associated_with": "associates",
    "literature": "co-mentioned", "animal_model": "animal model",
    "rna_expression": "expressed in", "somatic_mutation": "mutated in",
}


def rel_label(et_str):
    rel = et_str.split("::")[1]
    rev = rel.startswith("rev_")
    base = rel[4:] if rev else rel
    lab = REL_ABBR.get(base, base.replace("_", " "))
    return ("← " if rev else "") + lab   # left-arrow hint for reverse


def draw_node(ax, x, y, ntype, name, r=0.34):
    fill, edge, glyph = TYPE_STYLE.get(ntype, ("#eeeeee", "#888888", ntype))
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge, lw=1.8, zorder=3))
    ax.text(x, y, glyph, ha="center", va="center", fontsize=7, color=edge,
            fontweight="bold", zorder=4)
    ax.text(x, y - r - 0.14, name, ha="center", va="top", fontsize=7.5, zorder=4)


def draw_edge(ax, x0, x1, y, label, score=None, reverse=False):
    # Path always flows left->right. A `reverse` edge means the underlying graph
    # relation points right->left (we traverse it backwards); draw it dashed so
    # the direction of the actual edge is unambiguous.
    r = 0.34
    a = FancyArrowPatch((x0 + r, y), (x1 - r, y), arrowstyle="-|>",
                        mutation_scale=12, lw=1.3, color="#444444", zorder=2,
                        linestyle="--" if reverse else "-")
    ax.add_patch(a)
    mid = (x0 + x1) / 2
    ax.text(mid, y + 0.12, label, ha="center", va="bottom", fontsize=7,
            style="italic", color="#333333", zorder=4)
    if score is not None:
        ax.text(mid, y - 0.12, score, ha="center", va="top", fontsize=6.5,
                color="#888888", zorder=4)


def plot_paths(paths, node_meta, title, out_png, dx=2.2, dy=1.6):
    """paths: list of edge-lists; each edge = (et_str, src_tok, dst_tok, [s,n]).
    node_meta: token -> (ntype, name)."""
    nrows = len(paths)
    maxlen = max(len(p) for p in paths)
    fig_w = 1.2 + dx * maxlen
    fig_h = 1.0 + dy * nrows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    for ri, edges in enumerate(paths):
        y = (nrows - 1 - ri) * dy
        # nodes: source of edge0, then dst of each edge
        toks = [edges[0][1]] + [e[2] for e in edges]
        xs = [i * dx for i in range(len(toks))]
        for x, tok in zip(xs, toks):
            nt, nm = node_meta.get(tok, ("?", tok))
            draw_node(ax, x, y, nt, nm)
        for i, e in enumerate(edges):
            et = e[0]
            # Show the relation name only, not the per-edge s/n graph attributes.
            # The figure's job is the mechanistic ROUTING (what connects to what);
            # a path is only as strong as its object, and printing every edge's
            # score invites reading a path as weak because a structural connector
            # (shared-function, subtype-of) carries s~0, even when the anchor
            # target-disease edge is strong. Evidence strength over time is shown
            # in the temporal figure instead.
            is_rev = et.split("::")[1].startswith("rev_")
            draw_edge(ax, xs[i], xs[i + 1], y, rel_label(et), None, reverse=is_rev)
        ax.text(-0.9, y, f"P{ri}", ha="right", va="center", fontsize=8,
                fontweight="bold", color="#666")
    ax.set_xlim(-1.3, (maxlen) * dx + 0.4)
    ax.set_ylim(-0.9, (nrows - 1) * dy + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    # legend of node types
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", markerfacecolor=f, markeredgecolor=e,
                  markersize=9, label=g) for (f, e, g) in TYPE_STYLE.values()]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=len(leg), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_png)


if __name__ == "__main__":
    # Rebuild node_meta + scored paths for the biobridge candidates from the
    # saved JSON + graph (score/novelty lookup with <= visibility, reverse-aware).
    import torch, yaml
    N = json.load(open("/gpfs/scratch/bty414/biobridge_names.json"))
    GENE, DIS = N["genes"], N["dis"]
    P = json.load(open("/gpfs/scratch/bty414/biobridge_paths_totalcost.json"))
    cfg = yaml.safe_load(open("/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/"
                              "lr_grouped_k100_latest/lrgrpk100lat_s1/config.yaml"))
    data = torch.load(cfg["data"]["graph_file"], weights_only=False)
    m = torch.load(cfg["data"]["mappings_file"], weights_only=False)["node_mapping"]
    i2 = {k: {v: kk for kk, v in m[k].items()} for k in m}
    DEC = {"ENSG00000120217|EFO_0000588": (2016, "CD274 (PD-L1) → mesothelioma", "CD274_meso"),
           "ENSG00000112116|EFO_0003778": (2016, "IL17F → psoriatic arthritis", "IL17F_psa"),
           "ENSG00000181847|EFO_0003060": (2018, "TIGIT → non-small cell lung carcinoma", "TIGIT_nsclc")}

    def gid(tok):
        nt, idx = tok.split("#"); return nt, int(idx)

    def name_of(tok):
        nt, idx = gid(tok); acc = i2.get(nt, {}).get(idx, f"{nt}{idx}")
        if nt == "target": return GENE.get(acc, acc)
        if nt == "disease": return DIS.get(acc, acc)
        if nt == "molecule":
            return {"CHEMBL4297517": "mavacamten"}.get(acc, acc)
        return acc

    def sc_at(et, sg, dg, dy):
        st, rel, dt = et.split("::")
        if rel.startswith("rev_"):
            fwd = (dt, rel[4:], st); a, b = dg, sg
        else:
            fwd = (st, rel, dt); a, b = sg, dg
        if fwd not in data.edge_types: return None
        ei = data[fwd].edge_index; tm = data[fwd].edge_time; ea = data[fwd].edge_attr
        mask = (ei[0] == a) & (ei[1] == b)
        rows = sorted([(int(tm[i]), ea[i, 0].item(), ea[i, 1].item())
                       for i in mask.nonzero().flatten().tolist()])
        vis = [r for r in rows if r[0] <= dy]
        return (vis[-1][1], vis[-1][2]) if vis else None

    def is_trivial_trial_1hop(p):
        """A single-edge path whose only edge is a clinical_trial relation is the
        near-tautological direct query edge (it just restates target->disease via
        a trial edge). PaGE-Link exists to route around such shortcuts, so drop
        these from the displayed explanation; genuine 1-hop mechanistic edges are
        kept."""
        if p["n"] != 1:
            return False
        rel = p["E"][0][0].split("::")[1]
        base = rel[4:] if rel.startswith("rev_") else rel
        return base.startswith("clinical_trial")

    for key, ps in P.items():
        dy, title, tag = DEC[key]
        node_meta = {}
        plot_rows = []
        ps = [p for p in ps if not is_trivial_trial_1hop(p)]   # drop trial 1-hops
        for p in ps[:4]:                       # top-4 remaining paths
            edges = []
            for et, a, b, me in p["E"]:
                node_meta[a] = (gid(a)[0], name_of(a))
                node_meta[b] = (gid(b)[0], name_of(b))
                _, gs = gid(a); _, gd = gid(b)
                edges.append((et, a, b, sc_at(et, gs, gd, dy)))
            plot_rows.append(edges)
        plot_paths(plot_rows, node_meta, f"{title}  (decision {dy})",
                   f"{OUT}/explanation_paths_{tag}.png")
