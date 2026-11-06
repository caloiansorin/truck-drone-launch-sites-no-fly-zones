import itertools
import math
from dataclasses import dataclass, field

import numpy as np

INF = np.inf


@dataclass
class DateDP:
    tT: np.ndarray
    tD: np.ndarray
    cerere: np.ndarray
    n: int
    SL: float
    SR: float
    t_lansare: float
    t_recuperare: float
    P0: float
    Pm: np.ndarray
    B: float
    hover: bool
    eligibil: np.ndarray

    @property
    def nV(self):
        return self.tT.shape[0]


def date_din_scenariu(tT, tD, cerere, sc):
    n = len(cerere)
    if sc.model_energie == "payload":
        Pm = np.array([sc.putere_W(m) for m in cerere])
    else:
        Pm = np.full(n, sc.putere_W(0.0))
    return DateDP(
        tT=np.asarray(tT, float), tD=np.asarray(tD, float), cerere=np.asarray(cerere, float), n=n,
        SL=sc.SL_min, SR=sc.SR_min,
        t_lansare=sc.factor_putere_ls * sc.SL_min if sc.lansare_in_energie else 0.0,
        t_recuperare=sc.factor_putere_ls * sc.SR_min,
        P0=sc.putere_W(0.0), Pm=Pm, B=sc.baterie_Wh, hover=sc.hover_in_asteptare,
        eligibil=np.asarray(cerere) <= sc.Q_max_kg,
    )


def energie(D, d, a, b, t_cam):
    zbor = a + b
    asteptare = np.maximum(t_cam - zbor, 0.0) if D.hover else 0.0
    return (D.Pm[d] * (D.t_lansare + a) + D.P0 * (b + asteptare + D.t_recuperare)) / 60.0


def _biti(S, n):
    return [u for u in range(n) if S >> u & 1]


def drumuri_camion(D):
    n, nV = D.n, D.nV
    tT = D.tT
    C = np.arange(1, n + 1)
    H = np.full((1 << n, nV, n), INF)
    T = np.empty((1 << n, nV, nV))
    T[0] = tT
    for u in range(n):
        H[1 << u][:, u] = tT[:, C[u]]
    for S in range(1, 1 << n):
        membri = _biti(S, n)
        if len(membri) > 1:
            for u in membri:
                prev = S & ~(1 << u)
                H[S][:, u] = np.min(H[prev] + tT[C, C[u]][None, :], axis=1)
        T[S] = np.min(H[S][:, membri, None] + tT[C[membri]][None, :, :], axis=1)
    return T


def operatii_drona(D, T):
    n, nV = D.n, D.nV
    C = np.arange(1, n + 1)
    OP = np.full((1 << n, nV, nV), INF)
    ARG = np.full((1 << n, nV, nV), -1, dtype=np.int8)
    for S in range(1, 1 << n):
        for d in _biti(S, n):
            if not D.eligibil[d]:
                continue
            Tm = T[S & ~(1 << d)]
            a = D.tD[:, C[d]][:, None]
            b = D.tD[C[d], :][None, :]
            durata = D.SL + np.maximum(Tm, a + b) + D.SR
            durata[energie(D, d, a, b, Tm) > D.B + 1e-9] = INF
            durata[C[d], :] = INF
            mai_bun = durata < OP[S]
            OP[S][mai_bun] = durata[mai_bun]
            ARG[S][mai_bun] = d
    return OP, ARG


@dataclass
class Rezultat:
    makespan: float
    pasi: list = field(default_factory=list)
    noduri_R_folosite: list = field(default_factory=list)


def rezolva(D, bucla=True, revizitare=True, tabele=False):
    n, nV = D.n, D.nV
    plin = (1 << n) - 1
    tT = D.tT
    T = drumuri_camion(D)
    OP, ARG = operatii_drona(D, T)
    opfez = [S for S in range(1, 1 << n) if np.isfinite(OP[S]).any()]

    F = np.full((1 << n, nV), INF)
    F[0][0] = 0.0
    PS = np.full((1 << n, nV), -1, dtype=np.int64)
    PV = np.zeros((1 << n, nV), dtype=np.int64)
    PO = np.zeros((1 << n, nV), dtype=np.int64)
    client = np.zeros(nV, dtype=bool)
    client[1:n + 1] = True
    idx = np.arange(nV)

    def actualizeaza(S_tinta, w_mask, val, S_ant, vs, S_op):
        mai_bun = w_mask & (val < F[S_tinta] - 1e-12)
        F[S_tinta][mai_bun] = val[mai_bun]
        PS[S_tinta][mai_bun] = S_ant
        PV[S_tinta][mai_bun] = vs[mai_bun]
        PO[S_tinta][mai_bun] = S_op

    for S in range(1 << n):
        fS = F[S]
        if not np.isfinite(fS).any():
            continue
        servit = np.zeros(nV, dtype=bool)
        servit[1 + np.array(_biti(S, n), dtype=int)] = True

        dest = ~client & (idx != 0) if not revizitare else ~client | servit
        if not revizitare and S == plin:
            dest = dest | (idx == 0)
        cand = fS[:, None] + tT
        vs = np.argmin(cand, axis=0)
        actualizeaza(S, dest, cand[vs, idx], S, vs, 0)
        fS = F[S]
        activ = np.flatnonzero(np.isfinite(fS))

        for u in range(n):
            if not S >> u & 1:
                w = u + 1
                v = activ[np.argmin(fS[activ] + tT[activ, w])]
                val = fS[v] + tT[v, w]
                S2 = S | 1 << u
                if val < F[S2][w] - 1e-12:
                    F[S2][w] = val
                    PS[S2][w], PV[S2][w], PO[S2][w] = S, v, 0

        liber = plin & ~S
        for Sp in opfez:
            if Sp & S:
                continue
            cand = fS[activ][:, None] + OP[Sp][activ]
            if not bucla:
                cand = cand.copy()
                for r, v in enumerate(activ):
                    if not (v == 0 and S == 0 and (S | Sp) == plin):
                        cand[r, v] = INF
            r_min = np.argmin(cand, axis=0)
            val = cand[r_min, idx]
            vs = activ[r_min]
            U = S | Sp
            servitU = np.zeros(nV, dtype=bool)
            servitU[1 + np.array(_biti(U, n), dtype=int)] = True
            dest = ~client if not revizitare else ~client | servitU
            if not revizitare:
                dest = dest & ((idx != 0) | (U == plin))
            actualizeaza(U, dest, val, S, vs, Sp)
            for u in range(n):
                if liber >> u & 1 and not Sp >> u & 1:
                    w = u + 1
                    U2 = U | 1 << u
                    if val[w] < F[U2][w] - 1e-12:
                        F[U2][w] = val[w]
                        PS[U2][w], PV[U2][w], PO[U2][w] = S, vs[w], Sp

    rez = Rezultat(float(F[plin][0]))
    if tabele:
        rez.F, rez.OP, rez.opfez = F, OP, opfez
    if not math.isfinite(rez.makespan):
        return rez
    S, w = plin, 0
    pasi = []
    while PS[S][w] >= 0:
        S_ant, v, S_op = int(PS[S][w]), int(PV[S][w]), int(PO[S][w])
        if S_op == 0:
            pasi.append(("camion", v, w))
        else:
            pasi.append(("drona", v, S_op, int(ARG[S_op][v][w]), w))
        S, w = S_ant, v
    assert S == 0 and w == 0, (S, w)
    rez.pasi = pasi[::-1]
    rez.noduri_R_folosite = sorted({x for p in rez.pasi if p[0] == "drona"
                                    for x in (p[1], p[-1]) if x > n})
    return rez


def cost_ramas(D, rez):
    n, nV = D.n, D.nV
    plin = (1 << n) - 1
    tT, OP, opfez = D.tT, rez.OP, rez.opfez
    client = np.zeros(nV, dtype=bool)
    client[1:n + 1] = True
    G = np.full((1 << n, nV), INF)
    for S in range(plin, -1, -1):
        G1 = np.full(nV, INF)
        if S == plin:
            G1[0] = 0.0
        else:
            liber = plin & ~S
            for u in range(n):
                if liber >> u & 1:
                    G1 = np.minimum(G1, tT[:, u + 1] + G[S | 1 << u][u + 1])
            for Sp in opfez:
                if Sp & S:
                    continue
                U = S | Sp
                tinta = G[U].copy()
                for u in range(n):
                    if not U >> u & 1:
                        tinta[u + 1] = G[U | 1 << u][u + 1]
                G1 = np.minimum(G1, np.min(OP[Sp] + tinta[None, :], axis=1))
        servit = np.zeros(nV, dtype=bool)
        servit[1 + np.array(_biti(S, n), dtype=int)] = True
        dest = ~client | servit
        pas = np.min(tT[:, dest] + G1[dest][None, :], axis=1)
        G[S] = np.minimum(G1, pas)
    return G


def limita_prin_nod(rez, G):
    return np.min(rez.F + G, axis=0)


def verifica(D, rez):
    n = D.n
    servit = set()
    total = 0.0
    for p in rez.pasi:
        if p[0] == "camion":
            _, v, w = p
            total += D.tT[v][w]
            if 1 <= w <= n:
                servit.add(w - 1)
        else:
            _, v, Sp, d, w = p
            camion = [u for u in _biti(Sp, n) if u != d]
            best = INF
            for perm in itertools.permutations(camion):
                drum = [v, *(u + 1 for u in perm), w]
                best = min(best, sum(D.tT[x][y] for x, y in zip(drum, drum[1:])))
            a, b = D.tD[v][d + 1], D.tD[d + 1][w]
            assert D.eligibil[d] and v != d + 1
            assert energie(D, d, a, b, best) <= D.B + 1e-6, "energie depasita"
            total += D.SL + max(best, a + b) + D.SR
            servit |= set(_biti(Sp, n))
            if 1 <= w <= n:
                servit.add(w - 1)
    assert servit == set(range(n)), "nu toti clientii sunt serviti"
    assert abs(total - rez.makespan) < 1e-6, (total, rez.makespan)
    return total
