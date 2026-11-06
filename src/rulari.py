import csv
import sys
import time
import traceback
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

RAD = Path(__file__).resolve().parent.parent
V_REF = 50.0


def sarcini(set_):
    from parametri_operationali import BAZA_REALIST, configuratii
    from instanta_realista import CONFIGURATII
    b = BAZA_REALIST
    if set_ == "sensibilitate":
        sc = {"baza": (b, False),
              "v50": (b.cu(alpha=50 / V_REF), False), "v90": (b.cu(alpha=90 / V_REF), False),
              "SL2": (b.cu(SL_km=2 * V_REF / 60), False), "SL3": (b.cu(SL_km=3 * V_REF / 60), False),
              "AD10": (b.cu(autonomie_km=10.0), False), "AD30": (b.cu(autonomie_km=30.0), False),
              "centrale": (b, True)}
        return [(set_, nume, op, cen, cfg, seed, 2000)
                for nume, (op, cen) in sc.items() for cfg in CONFIGURATII for seed in range(2)]
    if set_ == "mdpi":
        return [(set_, op.eticheta, op, False, cfg, 0, 2000)
                for op in configuratii() for cfg in CONFIGURATII]
    raise ValueError(set_)


@lru_cache(maxsize=2)
def _instanta(centrale):
    from instanta_realista import construieste
    from tur_camion import tur_camion
    I0 = construieste(centrale=centrale)
    return I0, tur_camion(I0.T[: I0.n + 1, : I0.n + 1])


def lucreaza(sarcina):
    set_, nume, op, centrale, cfg, seed, iteratii = sarcina
    r = dict(set=set_, scenariu=nume, config=cfg, seed=seed, v_drona_kmh=op.v_drona_kmh,
             SL_min=op.SL_min, autonomie_km=op.autonomie_km, centrale=centrale)
    try:
        from euristica import Euristica
        from split_dp import ils_split, ordine_din_solutie
        from experiment import capete
        I0, tur = _instanta(centrale)
        I = I0.restrange(cfg)
        eu = Euristica(I, op)
        m_c = eu.makespan(eu.solutie_camion(tur))
        t0 = time.perf_counter()
        sg = eu.inserare_lacoma(eu.solutie_camion(tur))
        eu.verifica(sg)
        m_g = eu.makespan(sg)
        s, m = ils_split(eu, ordine_din_solutie(sg, I.n), iteratii=iteratii, seed=seed)
        eu.verifica(s)
        cap = capete(I, s)
        zbor = sum(I.Dkm[dict(s.R)[ua], c] + I.Dkm[c, dict(s.R)[ub]]
                   for ua, (ub, c) in s.misiuni.items())
        r.update(camion_min=m_c, lacom_min=m_g, split_ils_min=m,
                 castig_lacom_pct=100 * (1 - m_g / m_c), castig_pct=100 * (1 - m / m_c),
                 misiuni=len(s.misiuni), km_zbor=zbor,
                 **{f"capete_{k}": cap.get(k, 0) for k in
                    ("depozit", "client", "parcare", "refugiu", "acostament")},
                 timp_s=time.perf_counter() - t0, eroare="")
    except Exception:
        r["eroare"] = traceback.format_exc(limit=3).replace("\n", " | ")
    return r


def main(set_="sensibilitate", procese=6):
    iesire = RAD / "results" / ("sensibilitate.csv" if set_ == "sensibilitate" else "grila_mdpi.csv")
    lista = sarcini(set_)
    campuri = list(lucreaza.__code__.co_varnames[:0]) or None
    gata = set()
    if iesire.exists():
        with open(iesire) as f:
            gata = {(r["scenariu"], r["config"], r["seed"]) for r in csv.DictReader(f)
                    if not r["eroare"]}
    lista = [t for t in lista if (t[1], t[4], str(t[5])) not in gata]
    nou = not iesire.exists()
    with open(iesire, "a", newline="") as f, Pool(procese) as pool:
        w = None
        for r in pool.imap_unordered(lucreaza, lista):
            if w is None:
                campuri = ["set", "scenariu", "config", "seed", "v_drona_kmh", "SL_min",
                           "autonomie_km", "centrale", "camion_min", "lacom_min", "split_ils_min",
                           "castig_lacom_pct", "castig_pct", "misiuni", "km_zbor",
                           "capete_depozit", "capete_client", "capete_parcare", "capete_refugiu",
                           "capete_acostament", "timp_s", "eroare"]
                w = csv.DictWriter(f, fieldnames=campuri)
                if nou:
                    w.writeheader()
            w.writerow({k: r.get(k, "") for k in campuri})
            f.flush()
            print(r["scenariu"], r["config"], r["seed"],
                  round(r.get("castig_pct", float("nan")), 2), "EROARE" if r["eroare"] else "",
                  flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "sensibilitate", int(a[1]) if len(a) > 1 else 6)
