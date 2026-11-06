import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAD = Path(__file__).resolve().parent.parent
OUT = RAD / "results" / "figuri"
OUT.mkdir(parents=True, exist_ok=True)
CLIENTI = RAD / "data" / "customers.csv"

C_CFG = {"clienti": "#2a78d6", "parcari": "#eb6834", "toate": "#1baf7a"}
ET_CFG = {"clienti": "C: customers only", "parcari": "P: + parking / rest areas",
          "toate": "S: + road shoulders"}
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": TXT2,
    "axes.labelcolor": TXT, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def salveaza(fig, nume):
    fig.savefig(OUT / f"{nume}.pdf")
    fig.savefig(OUT / f"{nume}.png")
    plt.close(fig)
    print("scris", nume)


def scara(ax, km, x0f=0.05, y0f=0.04):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x, y = x0 + x0f * (x1 - x0), y0 + y0f * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color=TXT, lw=2.2, solid_capstyle="butt", zorder=8)
    for xx in (x, x + km * 1000):
        ax.plot([xx, xx], [y - 0.006 * (y1 - y0), y + 0.006 * (y1 - y0)], color=TXT, lw=1, zorder=8)
    ax.text(x + km * 500, y + 0.012 * (y1 - y0), f"{km:g} km", ha="center", va="bottom",
            fontsize=8, color=TXT, zorder=8)


def _solutie(cfg, iteratii=3000, seed=0):
    cale = RAD / "results" / (f"solutie_{cfg}.pkl" if seed == 0 else f"solutie_{cfg}_s{seed}.pkl")
    from instanta_realista import construieste
    from euristica import Euristica
    from split_dp import ils_split, ordine_din_solutie
    from tur_camion import tur_camion
    I0 = construieste()
    I = I0.restrange(cfg)
    eu = Euristica(I)
    if cale.exists():
        s = pickle.load(open(cale, "rb"))
    else:
        tur = tur_camion(I0.T[: I0.n + 1, : I0.n + 1])
        sg = eu.inserare_lacoma(eu.solutie_camion(tur))
        s, _ = ils_split(eu, ordine_din_solutie(sg, I.n), iteratii=iteratii, seed=seed)
        s._uid = None
        pickle.dump(s, open(cale, "wb"))
    eu.verifica(s)
    return I, eu, s


def _drum_rutier(R, a, b):
    from scipy.sparse.csgraph import dijkstra
    _, pred = dijkstra(R.A, directed=True, indices=[a], return_predecessors=True)
    p, cale = b, [b]
    while p != a and p >= 0:
        p = pred[0][p]
        cale.append(p)
    return cale[::-1]


def _seed_cel_mai_bun(cfg="toate"):
    f = RAD / "results" / "revizie_rulari.csv"
    if not f.exists():
        return 0
    d = pd.read_csv(f)
    d = d[(d.config == cfg) & d.eroare.isna()]
    return int(d.loc[d.split_ils_min.idxmin(), "seed"])


def fig_harta():
    import geopandas as gpd
    import shapely
    from retea_realista import construieste_retea
    from zone_restrictie import zone
    from puncte_lansare import judete, CRS_M

    I, eu, s = _solutie("toate", seed=_seed_cel_mai_bun())
    print("harta: makespan", round(eu.makespan(s), 1))
    R = construieste_retea()
    g = I.info["idx_graf"]
    to_m = lambda lon, lat: gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(CRS_M)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw=dict(width_ratios=[1.05, 1]))
    jud = judete()
    _, Z = zone()
    zone_u = shapely.unary_union(Z.geometry.values)
    noduri = [v for _, v in s.R]
    linii = []
    for a, b in zip(noduri, noduri[1:]):
        c = _drum_rutier(R, g[a], g[b])
        p = to_m(R.lon[c], R.lat[c])
        linii.append(np.column_stack([p.x, p.y]))
    pos = dict(s.R)
    sty = {"client": ("o", "#52514e", 9), "parcare": ("s", "#eb6834", 16),
           "refugiu": ("D", "#eb6834", 16), "acostament": ("^", "#1baf7a", 18)}
    xy = to_m(I.lon, I.lat)
    X, Y = xy.x.to_numpy(), xy.y.to_numpy()

    ordine_uid = {u: q for q, (u, _) in enumerate(s.R)}
    misiuni = sorted(s.misiuni.items(), key=lambda kv: ordine_uid[kv[0]])
    for ax_i, ax in enumerate(axs):
        gpd.GeoSeries([jud], crs=CRS_M).boundary.plot(ax=ax, color=TXT2, lw=0.6)
        gpd.GeoSeries([zone_u], crs=CRS_M).plot(ax=ax, color="#6250d6", alpha=0.15, lw=0)
        gpd.GeoSeries([zone_u], crs=CRS_M).boundary.plot(ax=ax, color="#6250d6", lw=0.5)
        for L in linii:
            ax.plot(L[:, 0], L[:, 1], color="#2a78d6", lw=1.0, zorder=2)
        for ua, (ub, c) in s.misiuni.items():
            a, b = pos[ua], pos[ub]
            if ax_i == 0:
                ax.plot([X[a], X[c], X[b]], [Y[a], Y[c], Y[b]], color="#e34948", lw=0.8,
                        ls=(0, (3, 2)), zorder=3)
            else:
                for p0, p1 in ((a, c), (c, b)):
                    an = ax.annotate("", xy=(X[p1], Y[p1]), xytext=(X[p0], Y[p0]), zorder=3,
                                     arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=1.1,
                                                     ls=(0, (3, 2)), mutation_scale=11,
                                                     shrinkA=3, shrinkB=4))
                    an.arrow_patch.set_clip_box(ax.bbox)
                    an.arrow_patch.set_clip_on(True)
        dr = [c for _, c in s.misiuni.values()]
        tr = [v for u, v in s.R if 1 <= v <= I.n and u not in s.trecere]
        mt = 10 if ax_i == 0 else 28
        ax.scatter(X[tr], Y[tr], s=mt, color="#2a78d6", zorder=4, linewidths=0)
        ax.scatter(X[dr], Y[dr], s=mt * 1.4, color="#e34948", zorder=4, linewidths=0)
        cap = {}
        for ua, (ub, c) in s.misiuni.items():
            for u in (ua, ub):
                cap[pos[u]] = I.sursa[pos[u]]
        for src, (mk, col, sz) in sty.items():
            if src == "client":
                continue
            k = [v for v, t in cap.items() if t == src]
            if k:
                ax.scatter(X[k], Y[k], marker=mk, s=sz if ax_i == 0 else sz * 3, facecolor="white",
                           edgecolor=col, lw=1.0 if ax_i == 0 else 1.6, zorder=5)
        ax.scatter(X[0], Y[0], marker="*", s=120, color=TXT, zorder=6)
        ax.set_axis_off()
    ac = np.array([v for v, t in cap.items() if t == "acostament"])
    k = np.argmax([np.sum(np.hypot(X[ac] - X[v], Y[ac] - Y[v]) < 15000) for v in ac])
    cx, cy, h = X[ac[k]], Y[ac[k]], 17000
    axs[1].set_xlim(cx - h, cx + h)
    axs[1].set_ylim(cy - h, cy + h)
    axs[0].add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, ec=TXT, lw=0.8))
    for q, (ua, (ub, c)) in enumerate(misiuni, 1):
        if abs(X[c] - cx) < h and abs(Y[c] - cy) < h:
            axs[1].annotate(str(q), (X[c], Y[c]), xytext=(5, 5), textcoords="offset points",
                            fontsize=8, color="#b8302f", fontweight="bold", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for L in linii:
        if len(L) < 6:
            continue
        m = len(L) // 2
        if abs(L[m, 0] - cx) < h and abs(L[m, 1] - cy) < h:
            axs[1].annotate("", xy=L[m + 1], xytext=L[m - 1], zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=1.2,
                                            mutation_scale=12))
    scara(axs[0], 20)
    scara(axs[1], 5)
    axs[0].set_title("(a) Bacău and Neamț counties", color=TXT, fontsize=10, loc="left")
    axs[1].set_title("(b) detail", color=TXT, fontsize=10, loc="left")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color="#2a78d6", lw=1.2, label="truck route (arrows: direction)"),
           Line2D([], [], color="#e34948", lw=0.9, ls=(0, (3, 2)),
                  label="drone sortie (in (b): launch → customer → recovery, numbered in route order)"),
           Line2D([], [], marker="o", ls="", color="#2a78d6", ms=4, label="truck customer"),
           Line2D([], [], marker="o", ls="", color="#e34948", ms=4, label="drone customer"),
           Line2D([], [], marker="^", ls="", mfc="white", mec="#1baf7a", ms=6,
                  label="launch/recovery: road shoulder"),
           Line2D([], [], marker="s", ls="", mfc="white", mec="#eb6834", ms=5,
                  label="launch/recovery: parking / rest area"),
           Line2D([], [], marker="*", ls="", color=TXT, ms=9, label="depot (Bacău)"),
           plt.Rectangle((0, 0), 1, 1, color="#6250d6", alpha=0.25,
                         label="no-fly zone (3 km around risk sites)")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, -0.09))
    fig.subplots_adjust(bottom=0.06, wspace=0.03)
    salveaza(fig, "fig_harta_solutie")


def fig_puncte():
    import geopandas as gpd
    import shapely
    from puncte_lansare import judete, CRS_M
    from zone_restrictie import zone
    P = pd.read_csv(RAD / "data" / "puncte_lansare.csv")
    P = P[~P.in_zona_restrictie]
    g = gpd.GeoDataFrame(P, geometry=gpd.points_from_xy(P.lon, P.lat), crs=4326).to_crs(CRS_M)
    jud = judete()
    _, Z = zone()
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    gpd.GeoSeries([jud], crs=CRS_M).boundary.plot(ax=ax, color=TXT2, lw=0.6)
    zu = shapely.unary_union(Z.geometry.values)
    gpd.GeoSeries([zu], crs=CRS_M).plot(ax=ax, color="#6250d6", alpha=0.18, lw=0)
    gpd.GeoSeries([zu], crs=CRS_M).boundary.plot(ax=ax, color="#6250d6", lw=0.5)
    sty = [("acostament", "^", "#0f8a5f", 16, "road shoulder"),
           ("parcare", "s", "#d9541e", 14, "parking lot"),
           ("refugiu", "D", "#1f5fb0", 26, "rest area / services")]
    for src, mk, col, sz, et in sty:
        q = g[g.sursa == src]
        ax.scatter(q.geometry.x, q.geometry.y, marker=mk, s=sz, color=col, lw=0,
                   label=f"{et} ({len(q)})", zorder=3)
    L = pd.read_csv(CLIENTI)
    lp = gpd.GeoSeries(gpd.points_from_xy(L.longitude, L.latitude), crs=4326).to_crs(CRS_M)
    ax.scatter(lp.x[1:], lp.y[1:], s=9, color=TXT, lw=0, label="customer locality (175)", zorder=4)
    ax.scatter(lp.x[:1], lp.y[:1], marker="*", s=140, color=TXT, lw=0, label="depot (Bacău)",
               zorder=5)
    ax.fill([], [], color="#6250d6", alpha=0.3, label="no-fly zone (3 km around 59 risk sites)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, markerscale=1.5,
              frameon=False)
    ax.set_axis_off()
    scara(ax, 20, x0f=0.62, y0f=0.04)
    salveaza(fig, "fig_puncte_lansare")


def _bare(ax, cats, valori, erori, culori, etichete):
    x = np.arange(len(cats))
    b = ax.bar(x, valori, width=0.55, color=culori, edgecolor="white", linewidth=2)
    if erori is not None:
        ax.errorbar(x, valori, yerr=erori, fmt="none", ecolor=TXT2, elinewidth=0.8, capsize=3)
    for xi, v in zip(x, valori):
        ax.text(xi, v + 0.35, f"{v:.1f}%", ha="center", va="bottom", color=TXT, fontsize=8)
    ax.set_xticks(x, etichete)
    return b


def fig_castig():
    f = RAD / "results" / "revizie_rulari.csv"
    d = pd.read_csv(f if f.exists() else RAD / "results" / "experiment_principal.csv")
    if "eroare" in d:
        d = d[d.eroare.isna()]
    nr = d.groupby("config").size().min()
    cfg = ["clienti", "parcari", "toate"]
    g = d.groupby("config")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(3)
    w = 0.36
    lac = [g.castig_lacom_pct.mean()[c] for c in cfg]
    spl = [g.castig_pct.mean()[c] for c in cfg]
    lo = [g.castig_pct.mean()[c] - g.castig_pct.min()[c] for c in cfg]
    hi = [g.castig_pct.max()[c] - g.castig_pct.mean()[c] for c in cfg]
    ax.bar(x - w / 2, lac, w, color="#86b6ef", edgecolor="white", lw=2, label="greedy insertion")
    ax.bar(x + w / 2, spl, w, color="#2a78d6", edgecolor="white", lw=2,
           label=f"split + ILS (mean of {nr} runs; bars: min–max; dots: runs)")
    ax.errorbar(x + w / 2, spl, yerr=[lo, hi], fmt="none", ecolor=TXT2, elinewidth=0.8, capsize=3)
    rng = np.random.default_rng(1)
    for k, c in enumerate(cfg):
        y = d[d.config == c].castig_pct.to_numpy()
        ax.scatter(k + w / 2 + rng.uniform(-0.08, 0.08, len(y)), y, s=9, color="white",
                   edgecolor=TXT, lw=0.6, zorder=4)
    for xi, v in zip(x - w / 2, lac):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=8, color=TXT)
    for xi, v, h in zip(x + w / 2, spl, hi):
        ax.text(xi, v + h + 0.3, f"{v:.1f}", ha="center", fontsize=8, color=TXT)
    ax.set_xticks(x, [ET_CFG[c] for c in cfg])
    ax.set_ylabel("makespan reduction vs. truck only (%)")
    ax.set_ylim(0, max(spl) * 1.25)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(axis="x", visible=False)
    salveaza(fig, "fig_castig_configuratii")


def fig_etalon():
    d = pd.read_csv(RAD / "results" / "etalon_dp_n8.csv")
    cfg = ["clienti", "parcari", "toate"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rng = np.random.default_rng(0)
    for k, c in enumerate(cfg):
        s = d[d.config == c]
        for off, col, nume, colr in ((-0.17, "gap_lacom_pct", "greedy insertion", "#86b6ef"),
                                     (0.17, "gap_pct", "split + ILS", "#2a78d6")):
            y = s[col].to_numpy()
            ax.scatter(k + off + rng.uniform(-0.06, 0.06, len(y)), y, s=16, color=colr,
                       edgecolor="white", lw=0.6, zorder=3, label=nume if k == 0 else None)
            ax.plot([k + off - 0.1, k + off + 0.1], [y.mean()] * 2, color=TXT, lw=1.2, zorder=4)
    ax.set_xticks(range(3), [ET_CFG[c] for c in cfg])
    ax.set_ylabel("gap to exact optimum (%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    salveaza(fig, "fig_etalon_dp")


def fig_sensibilitate():
    d = pd.read_csv(RAD / "results" / "sensibilitate.csv")
    d = d[d.eroare.isna()]
    m = d.groupby(["scenariu", "config"]).castig_pct.mean().unstack()
    grupe = [("drone speed (km/h)", [("v50", "50"), ("baza", "70"), ("v90", "90")]),
             ("launch = recovery time (min)", [("baza", "1"), ("SL2", "2"), ("SL3", "3")]),
             ("flight range per sortie (km)", [("AD10", "10"), ("baza", "20"), ("AD30", "30")]),
             ("no-fly zones", [("baza", "base"), ("centrale", "+ power\nplants")])]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True,
                            gridspec_kw=dict(width_ratios=[3, 3, 3, 2]))
    for ax, (titlu, sc) in zip(axs, grupe):
        x = np.arange(len(sc))
        w = 0.26
        for q, c in enumerate(["clienti", "parcari", "toate"]):
            v = [m.loc[s_, c] for s_, _ in sc]
            ax.bar(x + (q - 1) * w, v, w, color=C_CFG[c], edgecolor="white", lw=1.5,
                   label=ET_CFG[c])
            for xi, (s_, _) in zip(x, sc):
                y = d[(d.scenariu == s_) & (d.config == c)].castig_pct.to_numpy()
                ax.scatter([xi + (q - 1) * w] * len(y), y, s=8, color="white", edgecolor=TXT,
                           lw=0.6, zorder=4, label="individual run" if (q == 0 and xi == 0) else None)
        ax.set_xticks(x, [e for _, e in sc])
        ax.set_title(titlu, fontsize=9, color=TXT)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("makespan reduction vs. truck only (%)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    salveaza(fig, "fig_sensibilitate")


def fig_mdpi():
    d = pd.read_csv(RAD / "results" / "grila_mdpi.csv")
    d = d[d.eroare.isna()]
    d["alpha"] = (d.v_drona_kmh / 50).round(1)
    d["SL_km"] = (d.SL_min * 50 / 60).round(2)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for ax, (col, titlu) in zip(axs, [("SL_km", "launch = recovery penalty (km-eq.)"),
                                      ("alpha", "drone speed ratio α"),
                                      ("autonomie_km", "flight range AD (km)")]):
        g = d.groupby([col, "config"]).castig_pct.mean().unstack()
        for c in ["clienti", "parcari", "toate"]:
            ax.plot(g.index, g[c], color=C_CFG[c], lw=2, marker="o", ms=5, label=ET_CFG[c])
        ax.set_xticks(g.index)
        ax.set_title(titlu, fontsize=9, color=TXT)
    axs[0].set_ylabel("mean makespan reduction (%)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    salveaza(fig, "fig_grila_mdpi")


if __name__ == "__main__":
    ce = sys.argv[1] if len(sys.argv) > 1 else "toate"
    f = dict(puncte=fig_puncte, harta=fig_harta, castig=fig_castig, etalon=fig_etalon,
             sensibilitate=fig_sensibilitate, mdpi=fig_mdpi)
    for k, fn in f.items():
        if ce in (k, "toate"):
            fn()
