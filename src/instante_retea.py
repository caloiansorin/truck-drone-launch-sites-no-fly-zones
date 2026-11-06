import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse.csgraph import dijkstra

import retea_osm


@dataclass
class InstantaRetea:
    nume: str
    noduri_osm: list
    n: int
    id_localitati: list
    km_camion: np.ndarray
    km_drona: np.ndarray
    cerere: np.ndarray

    @property
    def nR(self):
        return len(self.noduri_osm) - 1 - self.n


@lru_cache(maxsize=1)
def _retea():
    G = retea_osm.incarca_graf()
    A, noduri, poz = retea_osm.matrice_adiacenta(G)
    lat = np.array([G.nodes[v]["y"] for v in noduri])
    lon = np.array([G.nodes[v]["x"] for v in noduri])
    grad = np.asarray((A > 0).sum(axis=0)).ravel() + np.asarray((A > 0).sum(axis=1)).ravel()
    loc = retea_osm.ataseaza_localitati(G, retea_osm.citeste_localitati())
    return A, noduri, poz, lat, lon, grad, loc


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _drum(pred, s, t):
    drum = [t]
    while drum[-1] != s:
        drum.append(pred[drum[-1]])
    return drum[::-1]


def _rarire(cand, ales, lat, lon, max_R, pas_min_km):
    R = []
    if not cand or max_R <= 0:
        return R
    cl, co = lat[cand], lon[cand]
    dmin = np.min([haversine_km(lat[b], lon[b], cl, co) for b in ales], axis=0)
    while len(R) < max_R:
        k = int(np.argmax(dmin))
        if dmin[k] < pas_min_km:
            break
        R.append(cand[k])
        dmin = np.minimum(dmin, haversine_km(lat[cand[k]], lon[cand[k]], cl, co))
    return R


def construieste(n=8, seed=0, max_R=40, pas_min_km=1.5, grad_min=3, cerere_max=5.0,
                 sursa_R="drum", max_R_zona=80, pas_zona_km=2.0, raza_zona_km=8.0):
    A, noduri, poz, lat, lon, grad, loc = _retea()
    rng = random.Random(seed)

    ids = list(loc.index)
    centru = rng.choice(ids)
    d = haversine_km(loc.loc[centru, "lat"], loc.loc[centru, "lon"],
                     loc["lat"].to_numpy(), loc["lon"].to_numpy())
    alese, vazute = [], set()
    for i in np.argsort(d):
        nod = poz[loc.iloc[i]["nod_osm"]]
        if nod not in vazute:
            vazute.add(nod)
            alese.append(ids[i])
        if len(alese) == n + 1:
            break
    rng.shuffle(alese)
    id_loc = alese
    baza = [poz[loc.loc[i, "nod_osm"]] for i in id_loc]

    cerere = np.array([round(rng.uniform(0, cerere_max), 2) for _ in range(n)])
    in_baza = set(baza)
    R = []
    if sursa_R in ("drum", "ambele"):
        _, pred = dijkstra(A, directed=True, indices=baza, return_predecessors=True)
        pe_drum = set()
        for a in range(len(baza)):
            for b in range(len(baza)):
                if a != b:
                    pe_drum.update(_drum(pred[a], baza[a], baza[b]))
        cand = sorted(v for v in pe_drum if grad[v] >= grad_min and v not in in_baza)
        R += _rarire(cand, baza, lat, lon, max_R, pas_min_km)
    if sursa_R in ("zona", "ambele"):
        dist = np.min([haversine_km(lat[b], lon[b], lat, lon) for b in baza], axis=0)
        cand = sorted(int(v) for v in np.flatnonzero((dist <= raza_zona_km) & (grad >= grad_min))
                      if v not in in_baza and v not in set(R))
        R += _rarire(cand, baza + R, lat, lon, max_R_zona, pas_zona_km)
    if sursa_R not in ("drum", "zona", "ambele"):
        raise ValueError(sursa_R)

    V = baza + R
    DV = dijkstra(A, directed=True, indices=V)[:, V] / 1000.0
    assert np.isfinite(DV).all(), "graful nu e tare conex pe nodurile instantei"
    km_drona = haversine_km(lat[V][:, None], lon[V][:, None], lat[V][None, :], lon[V][None, :])
    return InstantaRetea(f"OSM-n{n}-s{seed}", [noduri[v] for v in V], n, id_loc,
                         DV, km_drona, cerere)


def timpi(inst, sc, R=True):
    m = inst.n + 1 + (inst.nR if R else 0)
    tT = inst.km_camion[:m, :m] * 60.0 / sc.v_camion_kmh
    tD = inst.km_drona[:m, :m] * 60.0 / sc.v_drona_kmh
    return tT, tD
