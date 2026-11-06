from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from instante_retea import haversine_km
from retea_realista import construieste_retea

RAD = Path(__file__).resolve().parent.parent
CLIENTI_MDPI = RAD / "data" / "customers.csv"
CACHE = RAD / "cache" / "instante"
CONFIGURATII = ("clienti", "parcari", "toate")


@dataclass
class Instanta:
    nume: str
    n: int
    id_loc: list
    nume_loc: list
    cerere: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    sursa: list
    lansare_ok: np.ndarray
    T: np.ndarray
    Dkm: np.ndarray
    zbor_ok: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.lat)

    def restrange(self, config):
        surse = {"clienti": set(), "parcari": {"parcare", "refugiu"},
                 "toate": {"parcare", "refugiu", "acostament"}}[config]
        idx = [i for i in range(self.N) if i <= self.n or self.sursa[i] in surse]
        ix = np.ix_(idx, idx)
        return Instanta(f"{self.nume}-{config}", self.n, self.id_loc, self.nume_loc, self.cerere,
                        self.lat[idx], self.lon[idx], [self.sursa[i] for i in idx],
                        self.lansare_ok[idx], self.T[ix], self.Dkm[ix], self.zbor_ok[ix],
                        dict(self.info, config=config, idx_graf=self.info["idx_graf"][idx]))


def clienti_mdpi():
    df = pd.read_csv(CLIENTI_MDPI)
    loc = pd.read_csv(RAD / "data" / "localitati_noduri.csv").set_index("id")
    df["nod_osm"] = df.node_id.map(loc.nod_osm)
    dep = df[df.is_depot]
    return pd.concat([dep, df[~df.is_depot]], ignore_index=True)


def construieste(n=None, seed=0, raza_km=None, centrale=False, seed_cerere=None):
    from zone_restrictie import Restrictii

    nume = f"MDPI-{'all' if n is None else f'n{n}-s{seed}'}" + ("-centrale" if centrale else "")
    Rz = Restrictii(centrale=centrale)
    cale = CACHE / f"{nume}.npz"
    df = clienti_mdpi()
    if n is not None:
        rng = np.random.default_rng(seed)
        lat0, lon0 = df.latitude.to_numpy(), df.longitude.to_numpy()
        c0 = rng.integers(len(df))
        grup = np.argsort(haversine_km(lat0[c0], lon0[c0], lat0, lon0))[: n + 1]
        rng.shuffle(grup)
        df = df.iloc[grup].reset_index(drop=True)
        df["is_depot"] = [True] + [False] * n
    nn = len(df) - 1
    R = construieste_retea()
    P = R.puncte[~Rz.in_zona(R.puncte.lat.to_numpy(), R.puncte.lon.to_numpy())]
    raza_km = 30.0 if raza_km is None else raza_km
    d = haversine_km(df.latitude.to_numpy()[1:, None], df.longitude.to_numpy()[1:, None],
                     P.lat.to_numpy()[None, :], P.lon.to_numpy()[None, :])
    P = P[(d <= raza_km).any(axis=0)]

    lat = np.concatenate([df.latitude.to_numpy(), P.lat.to_numpy()])
    lon = np.concatenate([df.longitude.to_numpy(), P.lon.to_numpy()])
    sursa = ["depozit"] + ["client"] * nn + P.sursa.tolist()
    idx_graf = [R.poz[int(v)] for v in df.nod_osm] + P.idx.tolist()

    if cale.exists():
        z = np.load(cale)
        T, zbor_ok = z["T"], z["zbor_ok"]
    else:
        T = R.timpi(idx_graf, idx_graf)
        N = len(lat)
        i, j = np.triu_indices(N, 1)
        ok = Rz.zbor_permis(lat[i], lon[i], lat[j], lon[j])
        zbor_ok = np.eye(N, dtype=bool)
        zbor_ok[i, j] = zbor_ok[j, i] = ok
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cale, T=T, zbor_ok=zbor_ok)
    assert np.isfinite(T).all()

    in_zona = Rz.in_zona(lat, lon)
    Dkm = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    cerere = df.demand_kg.to_numpy()[1:].astype(float)
    if seed_cerere is not None:
        cerere = np.round(np.random.default_rng(seed_cerere).uniform(0.0, 5.0, nn), 2)
    return Instanta(nume, nn, df.node_id.tolist(), df.name.tolist(),
                    cerere, lat, lon, sursa, ~in_zona, T, Dkm, zbor_ok,
                    dict(raza_km=raza_km, centrale=centrale, idx_graf=np.asarray(idx_graf)))


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    I = construieste()
    print(f"{I.nume}: {I.n} clienti, {I.N - I.n - 1} puncte suplimentare "
          f"({pd.Series(I.sursa[I.n + 1:]).value_counts().to_dict()}), "
          f"{time.perf_counter() - t0:.0f} s")
    print(f"clienti in zone (doar camion): {(~I.lansare_ok[1:I.n + 1]).sum()}, "
          f"depozit in zona: {not I.lansare_ok[0]}, "
          f"clienti cu cerere <= 3 kg: {(I.cerere <= 3).sum()}")
    print(f"perechi de zbor blocate de zone: {(~I.zbor_ok).mean():.1%}")
    for c in CONFIGURATII:
        print(" ", c, I.restrange(c).N, "noduri")
