import collections
import sys
import time
from pathlib import Path

import pandas as pd

from euristica import Euristica
from instanta_realista import construieste, CONFIGURATII
from parametri_operationali import BAZA_REALIST
from split_dp import ils_split, ordine_din_solutie
from tur_camion import tur_camion

RAD = Path(__file__).resolve().parent.parent


def capete(I, s):
    R = dict(s.R)
    return collections.Counter(I.sursa[R[u]] for ua, (ub, _) in s.misiuni.items()
                               for u in (ua, ub))


def ruleaza(op=BAZA_REALIST, iteratii=3000, nr_seeds=3, eticheta="baza", verbos=True):
    I0 = construieste()
    tur = tur_camion(I0.T[: I0.n + 1, : I0.n + 1])
    rand = []
    for cfg in CONFIGURATII:
        I = I0.restrange(cfg)
        eu = Euristica(I, op)
        s0 = eu.solutie_camion(tur)
        m_c = eu.makespan(s0)
        t0 = time.perf_counter()
        sg = eu.inserare_lacoma(eu.solutie_camion(tur))
        eu.verifica(sg)
        m_g, t_g = eu.makespan(sg), time.perf_counter() - t0
        for seed in range(nr_seeds):
            t0 = time.perf_counter()
            s, m = ils_split(eu, ordine_din_solutie(sg, I.n), iteratii=iteratii, seed=seed)
            eu.verifica(s)
            cap = capete(I, s)
            r = dict(scenariu=eticheta, config=cfg, seed=seed, camion_min=m_c, lacom_min=m_g,
                     split_ils_min=m, castig_lacom_pct=100 * (1 - m_g / m_c),
                     castig_pct=100 * (1 - m / m_c), misiuni=len(s.misiuni),
                     clienti_drona_pct=100 * len(s.misiuni) / I.n,
                     **{f"capete_{k}": cap.get(k, 0) for k in
                        ("depozit", "client", "parcare", "refugiu", "acostament")},
                     timp_lacom_s=t_g, timp_s=time.perf_counter() - t0)
            rand.append(r)
            if verbos:
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()},
                      flush=True)
    return pd.DataFrame(rand)


def main(iteratii=3000, nr_seeds=3):
    df = ruleaza(iteratii=iteratii, nr_seeds=nr_seeds)
    df.to_csv(RAD / "results" / "experiment_principal.csv", index=False)
    print(df.groupby("config", sort=False)[["camion_min", "lacom_min", "split_ils_min",
                                             "castig_pct", "misiuni", "timp_s"]]
          .agg(["mean", "min"]).round(2).to_string())


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:]]
    main(*a)
