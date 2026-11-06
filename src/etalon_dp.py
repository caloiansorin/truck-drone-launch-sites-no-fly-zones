import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dp_exact import DateDP, rezolva, verifica
from euristica import Euristica
from split_dp import ils_split, ordine_din_solutie
from instanta_realista import construieste, CONFIGURATII
from parametri_operationali import BAZA_REALIST

RAD = Path(__file__).resolve().parent.parent
ITER_ILS = 500


def date_dp(I, op):
    v = op.v_drona_kmh / 60.0
    tD = I.Dkm / v
    tD = np.where(I.zbor_ok, tD, np.inf)
    tD[~I.lansare_ok, :] = np.inf
    tD[:, ~I.lansare_ok] = np.inf
    n = I.n
    elig = (I.cerere <= op.Q_max_kg) & I.lansare_ok[1: n + 1]
    return DateDP(tT=I.T, tD=tD, cerere=I.cerere, n=n, SL=op.SL_min, SR=op.SR_min,
                  t_lansare=0.0, t_recuperare=0.0, P0=60.0, Pm=np.full(n, 60.0),
                  B=op.autonomie_km / v, hover=True, eligibil=elig)


def main(n=8, nr_seeds=10, op=BAZA_REALIST):
    rand = []
    for seed in range(nr_seeds):
        I0 = construieste(n=n, seed=seed, raza_km=op.autonomie_km)
        for cfg in CONFIGURATII:
            I = I0.restrange(cfg)
            t0 = time.perf_counter()
            eu = Euristica(I, op)
            s = eu.solutie_camion()
            m_c = eu.makespan(s)
            s = eu.inserare_lacoma(s)
            m_g = eu.makespan(s)
            s, _ = ils_split(eu, ordine_din_solutie(s, I.n), iteratii=ITER_ILS, seed=seed)
            eu.verifica(s)
            m_h, t_h = eu.makespan(s), time.perf_counter() - t0
            t0 = time.perf_counter()
            D = date_dp(I, op)
            r = rezolva(D)
            verifica(D, r)
            t_dp = time.perf_counter() - t0
            rand.append(dict(instanta=I0.nume, config=cfg, noduri=I.N, camion=m_c,
                             lacom=m_g, gap_lacom_pct=None, euristica=m_h, optim=r.makespan,
                             gap_pct=100 * (m_h - r.makespan) / r.makespan,
                             misiuni_h=len(s.misiuni), misiuni_opt=sum(1 for p in r.pasi if p[0] != "camion"), timp_h_s=t_h, timp_dp_s=t_dp))
            rand[-1]["gap_lacom_pct"] = 100 * (m_g - r.makespan) / r.makespan
            print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in rand[-1].items()},
                  flush=True)
    df = pd.DataFrame(rand)
    df.to_csv(RAD / "results" / f"etalon_dp_n{n}.csv", index=False)
    print(df.groupby("config")[["gap_lacom_pct", "gap_pct", "timp_h_s", "timp_dp_s"]].agg(["mean", "max"]).round(2))


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
