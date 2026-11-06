from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import retea_osm

RAD = Path(__file__).resolve().parent.parent
PUNCTE = RAD / "data" / "puncte_lansare.csv"

VITEZA_PRACTICA = {
    "motorway": 110, "trunk": 90, "primary": 80, "secondary": 70, "tertiary": 60,
    "unclassified": 40, "residential": 30, "living_street": 15, "road": 30,
}
LIMITA_IMPLICITA = {"motorway": 130, "trunk": 100}
LIMITA_RO = {"RO:urban": 50, "RO:rural": 90, "RO:trunk": 100, "RO:motorway": 130,
             "RO:living_street": 20}
VITEZA_ACCES = 15.0
OCOLIRE_ACCES = 1.3


def _clasa(h):
    h = h[0] if isinstance(h, list) else h
    return h.removesuffix("_link")


def _limita(ms):
    vals = ms if isinstance(ms, list) else [ms]
    out = []
    for x in vals:
        x = str(x).strip()
        if x in LIMITA_RO:
            out.append(LIMITA_RO[x])
        elif x.split()[0].isdigit():
            out.append(float(x.split()[0]) * (1.609 if "mph" in x else 1.0))
    return min(out) if out else None


def viteza_arc(d, in_construit):
    c = _clasa(d.get("highway"))
    lim = _limita(d["maxspeed"]) if "maxspeed" in d else None
    if lim is None:
        lim = 50 if in_construit else LIMITA_IMPLICITA.get(c, 90)
    return min(lim, VITEZA_PRACTICA.get(c, 30))


@dataclass
class ReteaRealista:
    A: csr_matrix
    lat: np.ndarray
    lon: np.ndarray
    n_rutiere: int
    poz: dict
    puncte: pd.DataFrame

    def timpi(self, surse, tinte=None):
        T = dijkstra(self.A, directed=True, indices=list(surse))
        return T if tinte is None else T[:, list(tinte)]


@lru_cache(maxsize=1)
def construieste_retea():
    from puncte_lansare import zone_construite, CRS_M
    import geopandas as gpd

    G = retea_osm.incarca_graf()
    noduri = list(G.nodes)
    poz = {v: i for i, v in enumerate(noduri)}
    lat = [G.nodes[v]["y"] for v in noduri]
    lon = [G.nodes[v]["x"] for v in noduri]

    muchii = list(G.edges(data=True))
    mij = [d["geometry"].interpolate(0.5, normalized=True) if "geometry" in d else
           shapely.Point((G.nodes[u]["x"] + G.nodes[v]["x"]) / 2,
                         (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2) for u, v, d in muchii]
    construit = gpd.GeoSeries(mij, crs=4326).to_crs(CRS_M).within(zone_construite()).to_numpy()

    timp = {}
    for (u, v, d), ic in zip(muchii, construit):
        t = d["length"] / 1000.0 / viteza_arc(d, ic) * 60.0
        k = (poz[u], poz[v])
        timp[k] = min(timp.get(k, np.inf), t)

    P = pd.read_csv(PUNCTE)
    idx = []
    n0 = len(noduri)
    for r in P.itertuples():
        p = n0 + len(idx)
        idx.append(p)
        lat.append(r.lat)
        lon.append(r.lon)
        if r.sursa == "acostament":
            u, v = poz[int(r.u)], poz[int(r.v)]
            f = r.offset_m / r.lungime_arc_m
            if (u, v) in timp:
                t = timp[(u, v)]
                timp[(u, p)], timp[(p, v)] = f * t, (1 - f) * t
            if (v, u) in timp:
                t = timp[(v, u)]
                timp[(v, p)], timp[(p, u)] = (1 - f) * t, f * t
        else:
            q = poz[int(r.nod_osm)]
            t = r.dist_atasare_m / 1000.0 * OCOLIRE_ACCES / VITEZA_ACCES * 60.0
            timp[(q, p)] = timp[(p, q)] = t
    P = P.assign(idx=idx)
    N = len(lat)
    (rand, col), val = zip(*timp.keys()), list(timp.values())
    A = csr_matrix((val, (rand, col)), shape=(N, N))
    return ReteaRealista(A, np.array(lat), np.array(lon), n0, poz, P)


def main():
    R = construieste_retea()
    print(f"noduri: {R.n_rutiere} rutiere + {len(R.puncte)} puncte de lansare, arce {R.A.nnz}")

    dep = R.n_rutiere // 2
    T = R.timpi([dep], R.puncte.idx)[0]
    Tr = dijkstra(R.A.T.tocsr(), directed=True, indices=[dep])[0][R.puncte.idx]
    print("puncte inaccesibile (dus/intors):", int((~np.isfinite(T)).sum()),
          int((~np.isfinite(Tr)).sum()))

    loc = pd.read_csv(RAD / "data" / "localitati_noduri.csv").set_index("nume")
    perechi = [("Bacău", "Piatra Neamț"), ("Bacău", "Roman"), ("Bacău", "Onești"),
               ("Piatra Neamț", "Târgu Neamț"), ("Roman", "Piatra Neamț")]
    for a, b in perechi:
        ia = R.poz[loc[loc.index.str.startswith(a)].nod_osm.iloc[0]]
        ib = R.poz[loc[loc.index.str.startswith(b)].nod_osm.iloc[0]]
        print(f"  {a} -> {b}: {R.timpi([ia], [ib])[0, 0]:5.1f} min")


if __name__ == "__main__":
    main()
