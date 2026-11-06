import csv
import sys
import time
import traceback
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

RAD = Path(__file__).resolve().parent.parent

INVERSIUNI = [("clienti", "parcari", e) for e in
              ("a2-AD10-SL1", "a2-AD10-SL1.5", "a3-AD10-SL1", "a3-AD10-SL1.5", "a4-AD10-SL1",
               "a4-AD10-SL1.5")] + \
             [("parcari", "toate", e) for e in
              ("a3-AD20-SL1.5", "a3-AD30-SL1", "a4-AD30-SL0.5", "a4-AD30-SL1.5")]


def sarcini(set_):
    from parametri_operationali import BAZA_REALIST, configuratii
    from instanta_realista import CONFIGURATII
    b = BAZA_REALIST
    if set_ == "rulari":
        return [("baza", b, c, s, None, 20, 3000) for c in CONFIGURATII for s in range(10)]
    if set_ == "mase":
        return [(f"mase{m}", b, c, s, m, 20, 3000)
                for m in range(1, 11) for c in CONFIGURATII for s in range(3)]
    if set_ == "K":
        return [(f"K{k}", b, c, s, None, k, 3000)
                for k in (10, 20, 40) for c in CONFIGURATII for s in range(3)]
    if set_ == "inversiuni":
        ops = {o.eticheta: o for o in configuratii()}
        return [(e, ops[e], c, s, None, k, 2000)
                for a, bb, e in INVERSIUNI for c in (a, bb) for k in (20, 40) for s in range(3)]
    raise ValueError(set_)


@lru_cache(maxsize=2)
def _instanta(seed_cerere):
    from instanta_realista import construieste
    from tur_camion import tur_camion
    I0 = construieste(seed_cerere=seed_cerere)
    return I0, tur_camion(I0.T[: I0.n + 1, : I0.n + 1])


CAMPURI = ["set", "scenariu", "config", "seed", "seed_cerere", "K", "iteratii", "v_drona_kmh",
           "SL_min", "autonomie_km", "eligibili_drona", "camion_min", "lacom_min", "split_ils_min",
           "castig_lacom_pct", "castig_pct", "misiuni", "capete_client", "capete_parcare",
           "capete_refugiu", "capete_acostament", "timp_s", "eroare"]


def lucreaza(arg):
    set_, (nume, op, cfg, seed, seed_cerere, K, iteratii) = arg
    r = dict(set=set_, scenariu=nume, config=cfg, seed=seed, seed_cerere=seed_cerere, K=K,
             iteratii=iteratii, v_drona_kmh=op.v_drona_kmh, SL_min=op.SL_min,
             autonomie_km=op.autonomie_km)
    try:
        from euristica import Euristica
        from split_dp import ils_split, ordine_din_solutie
        from experiment import capete
        I0, tur = _instanta(seed_cerere)
        I = I0.restrange(cfg)
        eu = Euristica(I, op)
        m_c = eu.makespan(eu.solutie_camion(tur))
        sg = eu.inserare_lacoma(eu.solutie_camion(tur))
        eu.verifica(sg)
        m_g = eu.makespan(sg)
        t0 = time.perf_counter()
        s, m = ils_split(eu, ordine_din_solutie(sg, I.n), iteratii=iteratii, seed=seed, k_cand=K)
        eu.verifica(s)
        cap = capete(I, s)
        elig = int(((I.cerere <= op.Q_max_kg) & I.lansare_ok[1:I.n + 1]).sum())
        r.update(eligibili_drona=elig, camion_min=m_c, lacom_min=m_g, split_ils_min=m,
                 castig_lacom_pct=100 * (1 - m_g / m_c), castig_pct=100 * (1 - m / m_c),
                 misiuni=len(s.misiuni),
                 **{f"capete_{k}": cap.get(k, 0) for k in
                    ("client", "parcare", "refugiu", "acostament")},
                 timp_s=time.perf_counter() - t0, eroare="")
    except Exception:
        r["eroare"] = traceback.format_exc(limit=3).replace("\n", " | ")
    return r


def main(set_, procese=7):
    iesire = RAD / "results" / f"revizie_{set_}.csv"
    lista = sarcini(set_)
    gata = set()
    if iesire.exists():
        with open(iesire) as f:
            gata = {(r["scenariu"], r["config"], r["seed"], r["K"]) for r in csv.DictReader(f)
                    if not r["eroare"]}
    lista = [t for t in lista if (t[0], t[2], str(t[3]), str(t[5])) not in gata]
    nou = not iesire.exists()
    with open(iesire, "a", newline="") as f, Pool(procese) as pool:
        w = csv.DictWriter(f, fieldnames=CAMPURI)
        if nou:
            w.writeheader()
        for r in pool.imap_unordered(lucreaza, [(set_, t) for t in lista], chunksize=1):
            w.writerow({k: r.get(k, "") for k in CAMPURI})
            f.flush()
            print(set_, r["scenariu"], r["config"], r["seed"], r["K"],
                  round(r.get("castig_pct", float("nan")), 2), "EROARE " + r["eroare"] if r["eroare"] else "",
                  flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 7)
