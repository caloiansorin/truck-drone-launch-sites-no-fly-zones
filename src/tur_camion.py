import numpy as np


def cost_tur(T, tur):
    t = np.asarray(tur)
    return float(T[t, np.roll(t, -1)].sum())


def vecin_apropiat(T, start=0, rng=None, k=1):
    n = len(T)
    tur, liber = [start], np.ones(n, bool)
    liber[start] = False
    while liber.any():
        c = np.flatnonzero(liber)
        ordine = c[np.argsort(T[tur[-1], c])]
        nxt = ordine[rng.integers(min(k, len(ordine)))] if rng is not None else ordine[0]
        tur.append(int(nxt))
        liber[nxt] = False
    return tur


def _doi_opt(T, tur):
    n = len(tur)
    imbunatatit = True
    while imbunatatit:
        imbunatatit = False
        t = np.asarray(tur + [tur[0]])
        fw = np.concatenate([[0.0], np.cumsum(T[t[:-1], t[1:]])])
        bw = np.concatenate([[0.0], np.cumsum(T[t[1:], t[:-1]])])
        best, mut = -1e-9, None
        for i in range(0, n - 2):
            a, b = t[i], t[i + 1]
            j = np.arange(i + 2, n)
            c, d = t[j], t[j + 1]
            vechi = T[a, b] + (fw[j] - fw[i + 1]) + T[c, d]
            nou = T[a, c] + (bw[j] - bw[i + 1]) + T[b, d]
            delta = nou - vechi
            k = int(np.argmin(delta))
            if delta[k] < best:
                best, mut = delta[k], (i, j[k])
        if mut:
            i, j = mut
            tur = tur[: i + 1] + tur[i + 1: j + 1][::-1] + tur[j + 1:]
            imbunatatit = True
    return tur


def _or_opt(T, tur, lmax=3):
    n = len(tur)
    imbunatatit = True
    while imbunatatit:
        imbunatatit = False
        for L in range(1, lmax + 1):
            for s in range(1, n - L + 1):
                lant = tur[s: s + L]
                prev, nxt = tur[s - 1], tur[(s + L) % n]
                castig = T[prev, lant[0]] + T[lant[-1], nxt] - T[prev, nxt]
                rest = tur[:s] + tur[s + L:]
                r = np.asarray(rest + [rest[0]])
                cost_ins = T[r[:-1], lant[0]] + T[lant[-1], r[1:]] - T[r[:-1], r[1:]]
                k = int(np.argmin(cost_ins))
                if cost_ins[k] < castig - 1e-9:
                    tur = rest[: k + 1] + lant + rest[k + 1:]
                    imbunatatit = True
                    break
            if imbunatatit:
                break
    return tur


def cautare_locala(T, tur):
    while True:
        c0 = cost_tur(T, tur)
        tur = _or_opt(T, _doi_opt(T, tur))
        if cost_tur(T, tur) >= c0 - 1e-9:
            return tur


def _roteste(tur):
    i = tur.index(0)
    return tur[i:] + tur[:i]


def tur_camion(T, reporniri=20, seed=0):
    rng = np.random.default_rng(seed)
    best = cautare_locala(T, vecin_apropiat(T))
    for _ in range(reporniri):
        t = cautare_locala(T, vecin_apropiat(T, rng=rng, k=3))
        if cost_tur(T, t) < cost_tur(T, best):
            best = t
    return _roteste(best)


def main():
    import time
    import pandas as pd
    from instanta_realista import construieste, CLIENTI_MDPI

    I = construieste()
    T = I.T[: I.n + 1, : I.n + 1]
    t0 = time.perf_counter()
    tur = tur_camion(T)
    print(f"tur camion: {cost_tur(T, tur):.1f} min ({cost_tur(T, tur) / 60:.1f} h), "
          f"{time.perf_counter() - t0:.0f} s")

    rut = pd.read_csv(CLIENTI_MDPI.parent / "ruta_optima_tsp.txt", sep=r"\s+", header=None)
    poz = {v: i for i, v in enumerate(I.id_loc)}
    tur_mdpi = [poz[int(v)] for v in rut[0]]
    c_m = cost_tur(T, tur_mdpi)
    c_m_inv = cost_tur(T, tur_mdpi[::-1])
    print(f"turul MDPI pe timpii noi: {min(c_m, c_m_inv):.1f} min "
          f"(turul nostru e cu {100 * (1 - cost_tur(T, tur) / min(c_m, c_m_inv)):.2f}% mai scurt)")


if __name__ == "__main__":
    main()
