import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from parametri_operationali import BAZA_REALIST, Operatiuni
from tur_camion import tur_camion

K_PUNCTE = 8
EPS = 1e-7
PRAG_LUSTRU = 0.03


@dataclass
class Solutie:
    R: list
    misiuni: dict
    trecere: set = field(default_factory=set)
    _uid: itertools.count = field(default_factory=lambda: itertools.count(10 ** 6))

    def nou_uid(self):
        return next(self._uid)


class Euristica:
    def __init__(self, I, op: Operatiuni = BAZA_REALIST):
        self.I, self.op = I, op
        self.T, self.D = I.T, I.Dkm
        self.v = op.v_drona_kmh / 60.0
        self.E = op.autonomie_km / self.v
        self.SL, self.SR = op.SL_min, op.SR_min
        n = I.n
        self.eligibil = np.zeros(I.N, bool)
        self.eligibil[1: n + 1] = (I.cerere <= op.Q_max_kg) & I.lansare_ok[1: n + 1]
        self.puncte = np.array([p for p in range(I.N) if I.lansare_ok[p]], dtype=int)

    def zbor(self, a, c, b):
        return (self.D[a, c] + self.D[c, b]) / self.v

    def makespan(self, s: Solutie):
        R, M = s.R, s.misiuni
        poz = {u: k for k, (u, _) in enumerate(R)}
        t, k, facut = 0.0, 0, None
        while k < len(R) - 1:
            u = R[k][0]
            if u in M and facut != k:
                ub, c = M[u]
                q = poz[ub]
                if q < k:
                    return np.inf
                tau = sum(self.T[R[x][1], R[x + 1][1]] for x in range(k, q))
                f = self.zbor(R[k][1], c, R[q][1])
                t += self.SL + max(tau, f) + self.SR
                if q == k:
                    facut = k
                k = q
            else:
                t += self.T[R[k][1], R[k + 1][1]]
                k += 1
        return t

    def verifica(self, s: Solutie):
        poz = {u: k for k, (u, _) in enumerate(s.R)}
        acoperit = np.zeros(len(s.R), int)
        serviti = [v for u, v in s.R if 1 <= v <= self.I.n and u not in s.trecere]
        for ua, (ub, c) in s.misiuni.items():
            ka, kb = poz[ua], poz[ub]
            a, b = s.R[ka][1], s.R[kb][1]
            assert ka <= kb and self.eligibil[c]
            assert self.I.lansare_ok[a] and self.I.lansare_ok[b]
            assert self.I.zbor_ok[a, c] and self.I.zbor_ok[c, b]
            assert self.D[a, c] + self.D[c, b] <= self.op.autonomie_km + 1e-9
            tau = sum(self.T[s.R[x][1], s.R[x + 1][1]] for x in range(ka, kb))
            assert max(tau, self.zbor(a, c, b)) <= self.E + 1e-9
            acoperit[ka:kb] += 1
            serviti.append(c)
        assert acoperit.max(initial=0) <= 1, "misiuni suprapuse"
        assert sorted(serviti) == list(range(1, self.I.n + 1)), "clienti lipsa sau dublati"

    def fezabil(self, s: Solutie):
        R = s.R
        if R[0][1] != 0 or R[-1][1] != 0:
            return False
        poz = {u: k for k, (u, _) in enumerate(R)}
        interv = []
        for ua, (ub, c) in s.misiuni.items():
            ka, kb = poz[ua], poz[ub]
            if ka > kb:
                return False
            a, b = R[ka][1], R[kb][1]
            if not (self.I.lansare_ok[a] and self.I.lansare_ok[b] and self.I.zbor_ok[a, c]
                    and self.I.zbor_ok[c, b]):
                return False
            if self.D[a, c] + self.D[c, b] > self.op.autonomie_km + 1e-9:
                return False
            tau = sum(self.T[R[x][1], R[x + 1][1]] for x in range(ka, kb))
            if max(tau, self.zbor(a, c, b)) > self.E + 1e-9:
                return False
            interv.append((ka, kb))
        interv.sort()
        for (a1, b1), (a2, b2) in zip(interv, interv[1:]):
            if a2 < b1 or (a2 == b1 == a1) or (a2 == b2 == b1):
                return False
        return True

    def mutari_opriri(self, s: Solutie):
        m0 = self.makespan(s)
        imbunatatit = True
        while imbunatatit:
            imbunatatit = False
            L = len(s.R)
            for k in range(1, L - 1):
                rest = s.R[:k] + s.R[k + 1:]
                for q in range(1, L - 1):
                    if q == k:
                        continue
                    R2 = rest[:q] + [s.R[k]] + rest[q:]
                    s2 = Solutie(R2, s.misiuni, s.trecere, s._uid)
                    m = self.makespan(s2)
                    if m < m0 - 1e-6 and self.fezabil(s2):
                        s, m0, imbunatatit = s2, m, True
                        break
                if imbunatatit:
                    break
        return s

    def solutie_camion(self, tur=None):
        n = self.I.n
        if tur is None:
            tur = tur_camion(self.T[: n + 1, : n + 1])
        s = Solutie([], {})
        s.R = [(s.nou_uid(), v) for v in tur + [tur[0]]]
        return s

    def _candidati_puncte(self, u, w, c, sens):
        P = self.puncte[self.puncte != c]
        if not len(P):
            return P, np.zeros(0)
        ok = (self.I.zbor_ok[P, c] if sens == 0 else self.I.zbor_ok[c, P]) & \
             (self.D[P, c] <= self.op.autonomie_km)
        P = P[ok]
        if not len(P):
            return P, np.zeros(0)
        scor = self.T[u, P] + self.T[P, w] - self.T[u, w] + self.D[P, c] / self.v
        k = np.argsort(scor)[:K_PUNCTE]
        return P[k], scor[k]

    def cea_mai_buna_misiune(self, s: Solutie, c):
        kc = next(k for k, (u, v) in enumerate(s.R) if v == c and u not in s.trecere)
        uc = s.R[kc][0]
        if uc in s.misiuni or any(ub == uc for ub, _ in s.misiuni.values()):
            return np.inf, None
        R = s.R[:kc] + s.R[kc + 1:]
        s2 = Solutie(R, s.misiuni, s.trecere, s._uid)
        baza = self.makespan(s2) - self.makespan(s)
        nod = np.array([v for _, v in R])
        L = len(R)
        pre = np.concatenate([[0.0], np.cumsum(self.T[nod[:-1], nod[1:]])])
        poz = {u: k for k, (u, _) in enumerate(R)}
        acoperit = np.zeros(L - 1, bool)
        capat = np.zeros(L, bool)
        lanseaza = np.zeros(L, bool)
        for ua, (ub, _) in s.misiuni.items():
            acoperit[poz[ua]: poz[ub]] = True
            capat[poz[ua]] = capat[poz[ub]] = True
            lanseaza[poz[ua]] = True
        AD, D, T, Z, ok = self.op.autonomie_km, self.D, self.T, self.I.zbor_ok, self.I.lansare_ok
        aproape = D[nod, c] <= AD
        best = (np.inf, None)
        cache_pts = {}

        def pts(k, sens):
            if (k, sens) not in cache_pts:
                cache_pts[(k, sens)] = self._candidati_puncte(nod[k], nod[k + 1], c, sens)[0]
            return cache_pts[(k, sens)]

        for i in range(L - 1):
            if not (aproape[i] or aproape[i + 1]):
                continue
            for j in range(i, L):
                if j > i and acoperit[j - 1]:
                    break
                if pre[j] - pre[min(i + 1, j)] > self.E:
                    break
                vechi = pre[j] - pre[i]
                opt_a = []
                if ok[nod[i]] and Z[nod[i], c] and not lanseaza[i] and not (j == i and capat[i]):
                    opt_a.append(("s", np.array([nod[i]])))
                if j > i and not acoperit[i]:
                    opt_a.append(("p", pts(i, 0)))
                opt_b = []
                if ok[nod[j]] and Z[c, nod[j]]:
                    opt_b.append(("s", np.array([nod[j]])))
                if j > i:
                    opt_b.append(("p", pts(j - 1, 1)))
                for ta, A in opt_a:
                    for tb, B in opt_b:
                        if not len(A) or not len(B):
                            continue
                        if j == i and not (ta == "s" and tb == "s"):
                            continue
                        Ac, Bc = A[:, None], B[None, :]
                        if ta == "s" and tb == "s":
                            tau = np.full((1, 1), pre[j] - pre[i])
                            ext = 0.0
                        elif ta == "s":
                            tau = pre[j - 1] - pre[i] + T[nod[j - 1], Bc]
                            ext = T[Bc, nod[j]]
                        elif tb == "s":
                            tau = T[Ac, nod[i + 1]] + pre[j] - pre[i + 1]
                            ext = T[nod[i], Ac]
                        elif j == i + 1:
                            tau = T[Ac, Bc]
                            ext = T[nod[i], Ac] + T[Bc, nod[j]]
                        else:
                            tau = T[Ac, nod[i + 1]] + pre[j - 1] - pre[i + 1] + T[nod[j - 1], Bc]
                            ext = T[nod[i], Ac] + T[Bc, nod[j]]
                        dist = D[Ac, c] + D[c, Bc]
                        f = dist / self.v
                        aer = np.maximum(tau, f)
                        nou = ext + self.SL + aer + self.SR
                        nou = np.where((dist <= AD) & (aer <= self.E) & Z[Ac, c] & Z[c, Bc],
                                       nou, np.inf)
                        x, y = np.unravel_index(np.argmin(nou), nou.shape)
                        d = baza + nou[x, y] - vechi
                        if d < best[0]:
                            best = (d, (kc, i, j, ta, int(A[x]), tb, int(B[y]), c))
        return best

    def aplica(self, s: Solutie, mutare):
        kc, i, j, ta, a, tb, b, c = mutare
        R = s.R[:kc] + s.R[kc + 1:]
        ua = R[i][0] if ta == "s" else s.nou_uid()
        ub = R[j][0] if tb == "s" else s.nou_uid()
        nouR = list(R)
        if tb == "p":
            nouR.insert(j, (ub, b))
        if ta == "p":
            nouR.insert(i + 1, (ua, a))
        s.R = nouR
        s.misiuni = dict(s.misiuni)
        s.misiuni[ua] = (ub, c)
        s.trecere = set(s.trecere) | {u for u, t in ((ua, ta), (ub, tb)) if t == "p"}
        return s

    def inserare_lacoma(self, s: Solutie, verbos=False, rng=None, rcl=1):
        while True:
            servit_camion = [v for u, v in s.R if self.eligibil[v] and u not in s.trecere]
            cand = []
            for c in servit_camion:
                d, m = self.cea_mai_buna_misiune(s, c)
                if d < -EPS:
                    cand.append((d, m))
            if not cand:
                return s
            cand.sort(key=lambda x: x[0])
            best = cand[rng.integers(min(rcl, len(cand)))] if rng is not None else cand[0]
            m0 = self.makespan(s)
            s = self.aplica(s, best[1])
            m1 = self.makespan(s)
            assert abs(m1 - (m0 + best[0])) < 1e-6, (m0, best[0], m1)
            if verbos:
                print(f"  misiune {len(s.misiuni):3d}: client {best[1][-1]:3d}, "
                      f"delta {best[0]:7.2f} min -> {m1:8.1f}")


def ruineaza(eu: Euristica, s: Solutie, k, rng):
    if not s.misiuni or k <= 0:
        return s
    M = dict(s.misiuni)
    scoase = rng.choice(list(M), size=min(k, len(M)), replace=False)
    clienti = [M.pop(u)[1] for u in scoase]
    capete = set(M) | {ub for ub, _ in M.values()}
    R = [(u, v) for u, v in s.R if u not in s.trecere or u in capete]
    s = Solutie(R, M, {u for u in s.trecere if u in capete}, s._uid)
    for c in clienti:
        best = (np.inf, None)
        for k2 in range(len(s.R) - 1):
            cand = Solutie(s.R[: k2 + 1] + [(s.nou_uid(), c)] + s.R[k2 + 1:], s.misiuni,
                           s.trecere, s._uid)
            m = eu.makespan(cand)
            if m < best[0]:
                best = (m, cand)
        s = best[1]
    return s


def _cautare_pe_submultime(T, noduri, tur_init):
    from tur_camion import cautare_locala
    poz = {v: k for k, v in enumerate(noduri)}
    sub = T[np.ix_(noduri, noduri)]
    t = cautare_locala(sub, [poz[v] for v in tur_init])
    t = [noduri[k] for k in t]
    i = t.index(0)
    return t[i:] + t[:i]


def _inserare_ieftina(T, tur, clienti):
    tur = list(tur)
    for c in clienti:
        r = np.asarray(tur + [tur[0]])
        cost = T[r[:-1], c] + T[c, r[1:]] - T[r[:-1], r[1:]]
        k = int(np.argmin(cost))
        tur.insert(k + 1, c)
    return tur


def _double_bridge(tur, rng):
    n = len(tur)
    if n < 8:
        return list(tur)
    a, b, c = sorted(rng.choice(np.arange(1, n), size=3, replace=False))
    return tur[:a] + tur[b:c] + tur[a:b] + tur[c:]


def ils(eu: Euristica, iteratii=30, seed=0, tur=None, verbos=False):
    rng = np.random.default_rng(seed)
    T = eu.T
    best = eu.mutari_opriri(eu.inserare_lacoma(eu.solutie_camion(tur)))
    m_best = eu.makespan(best)

    def lustruieste(s, m):
        if m < m_best * (1 + PRAG_LUSTRU):
            s = eu.mutari_opriri(s)
            m = eu.makespan(s)
        return s, m
    for it in range(iteratii):
        if rng.random() < 0.6:
            k = int(rng.integers(1, max(2, len(best.misiuni) // 2 + 1)))
            s = ruineaza(eu, best, k, rng)
            s = eu.inserare_lacoma(s, rng=rng, rcl=int(rng.integers(1, 4)))
            s, m = lustruieste(s, eu.makespan(s))
            if m < m_best - EPS:
                best, m_best = s, m
                if verbos:
                    print(f"  ils {it:3d} (r&r): {m_best:.2f}", flush=True)
            continue
        drona = [c for _, c in best.misiuni.values()]
        tur_b = [v for u, v in best.R[:-1] if u not in best.trecere]
        rng.shuffle(drona)
        pastreaza = drona[: rng.integers(0, len(drona) + 1)] if rng.random() < 0.3 else []
        scoase = [c for c in drona if c not in pastreaza]
        baza = tur_b if not pastreaza else _inserare_ieftina(T, tur_b, pastreaza)
        if rng.random() < 0.5:
            baza = _double_bridge(baza, rng)
        baza = _cautare_pe_submultime(T, sorted(set(baza)), baza)
        tur_nou = _inserare_ieftina(T, baza, scoase)
        s = eu.inserare_lacoma(eu.solutie_camion(tur_nou), rng=rng, rcl=int(rng.integers(1, 3)))
        s, m = lustruieste(s, eu.makespan(s))
        if m < m_best - EPS:
            best, m_best = s, m
            if verbos:
                print(f"  ils {it:3d}: {m_best:.2f}", flush=True)
    return best


def rezumat(eu: Euristica, s: Solutie, m_camion):
    m = eu.makespan(s)
    surse = [eu.I.sursa[v] for ua, (ub, _) in s.misiuni.items()
             for v in (dict(s.R)[ua], dict(s.R)[ub])]
    import collections
    return dict(makespan_min=round(m, 1), camion_singur_min=round(m_camion, 1),
                castig_pct=round(100 * (1 - m / m_camion), 2), misiuni=len(s.misiuni),
                capete=dict(collections.Counter(surse)))


def main():
    from instanta_realista import construieste, CONFIGURATII
    I0 = construieste()
    tur = tur_camion(I0.T[: I0.n + 1, : I0.n + 1])
    for cfg in CONFIGURATII:
        I = I0.restrange(cfg)
        eu = Euristica(I)
        s = eu.solutie_camion(tur)
        m_c = eu.makespan(s)
        t0 = time.perf_counter()
        s = eu.inserare_lacoma(s)
        eu.verifica(s)
        print(cfg, rezumat(eu, s, m_c), f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
