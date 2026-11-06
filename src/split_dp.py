import time

import numpy as np

from euristica import Euristica, Solutie

K_CAND = 20
W_MAX = 6
SEG_MAX = 8


class Split:
    def __init__(self, eu: Euristica, k_cand=K_CAND, w_max=W_MAX):
        self.eu, self.I = eu, eu.I
        self.K, self.W = k_cand, w_max
        I = self.I
        ok = np.flatnonzero(I.lansare_ok)
        self.ok_noduri = ok
        self.cand = {}
        for v in range(I.n + 1):
            d = np.minimum(I.T[v, ok], I.T[ok, v])
            self.cand[v] = ok[np.argsort(d)[: self.K + 1]]

    def _C(self, v):
        c = self.cand[v]
        return c[c != v]

    def evalueaza(self, ordine, cu_drum=False):
        eu, I, T, D, Z = self.eu, self.I, self.I.T, self.I.Dkm, self.I.zbor_ok
        t = [0, *ordine, 0]
        n = len(ordine)
        loc = [None] * (n + 2)
        loc[0] = np.array([0])
        for k in range(1, n + 2):
            loc[k] = np.unique(np.concatenate([[t[k]], self._C(t[k])])) if k <= n else np.array([0])
        val = [np.full(len(l), np.inf) for l in loc]
        pred = [[None] * len(l) for l in loc]
        poz = [{int(x): q for q, x in enumerate(l)} for l in loc]
        val[0][0] = 0.0
        v, SL, SR, E, AD = eu.v, eu.SL, eu.SR, eu.E, eu.op.autonomie_km
        ok = I.lansare_ok
        for i in range(n + 1):
            Vi = val[i]
            if not np.isfinite(Vi).any():
                continue
            X = loc[i]
            fin = np.isfinite(Vi)
            dest = t[i + 1]
            c = Vi + T[X, dest]
            q = int(np.argmin(c))
            r = poz[i + 1][dest]
            if c[q] < val[i + 1][r]:
                val[i + 1][r] = c[q]
                pred[i + 1][r] = (i, q, None)
            A_extra = self._C(t[i]) if i > 0 else self._C(0)
            for k in range(i + 1, min(i + self.W, n) + 1):
                for j in range(i + 1, k + 1):
                    cj = t[j]
                    if not eu.eligibil[cj]:
                        continue
                    camion = [t[m] for m in range(i + 1, k + 1) if m != j]
                    interior = sum(T[x, y] for x, y in zip(camion, camion[1:]))
                    if interior > E:
                        continue
                    B = self._C(t[k]) if j == k else np.concatenate([[t[k]], self._C(t[k])])
                    B = B[ok[B] & Z[cj, B] & (D[cj, B] <= AD)]
                    if not len(B):
                        continue
                    A = np.unique(np.concatenate([X[fin], A_extra]))
                    A = A[ok[A] & Z[A, cj] & (D[A, cj] <= AD)]
                    if not len(A):
                        continue
                    if camion:
                        tau = T[A, camion[0]][:, None] + interior + T[camion[-1], B][None, :]
                    else:
                        tau = T[np.ix_(A, B)]
                    f = (D[A, cj][:, None] + D[cj, B][None, :]) / v
                    aer = np.maximum(tau, f)
                    cost = np.where((aer <= E + 1e-9) & (f * v <= AD + 1e-9), SL + aer + SR, np.inf)
                    if not np.isfinite(cost).any():
                        continue
                    intrare = Vi[:, None] + T[np.ix_(X, A)]
                    qx = np.argmin(intrare, axis=0)
                    intr = intrare[qx, np.arange(len(A))]
                    tot = intr[:, None] + cost
                    best_a = np.argmin(tot, axis=0)
                    tb = tot[best_a, np.arange(len(B))]
                    for ib in np.flatnonzero(np.isfinite(tb)):
                        b = int(B[ib])
                        if b not in poz[k]:
                            continue
                        r = poz[k][b]
                        if tb[ib] < val[k][r] - 1e-12:
                            ia = best_a[ib]
                            val[k][r] = tb[ib]
                            pred[k][r] = (i, int(qx[ia]), (int(A[ia]), cj, b, camion))
        m = float(val[n + 1][0])
        if not cu_drum:
            return m
        pasi, k, r = [], n + 1, 0
        while k > 0:
            i, q, mis = pred[k][r]
            pasi.append((int(loc[i][q]), t[k] if mis is None else None, mis))
            k, r = i, q
        return m, pasi[::-1]

    def solutie(self, ordine):
        m, pasi = self.evalueaza(ordine, cu_drum=True)
        s = Solutie([], {})
        s.R = [(s.nou_uid(), 0)]
        for x, dest, mis in pasi:
            if mis is None:
                s.R.append((s.nou_uid(), dest))
                continue
            a, cj, b, camion = mis
            if a != s.R[-1][1] or (not camion and a == b) or s.R[-1][0] in s.misiuni:
                ua = s.nou_uid()
                s.R.append((ua, a))
                s.trecere.add(ua)
            else:
                ua = s.R[-1][0]
            for c in camion:
                s.R.append((s.nou_uid(), c))
            if not camion and a == b:
                ub = ua
            elif camion and b == camion[-1]:
                ub = s.R[-1][0]
            else:
                ub = s.nou_uid()
                s.R.append((ub, b))
                s.trecere.add(ub)
            s.misiuni[ua] = (ub, cj)
        if s.R[-1][1] != 0:
            s.R.append((s.nou_uid(), 0))
        return s, m


def ordine_din_solutie(s: Solutie, n):
    o = []
    for u, v in s.R:
        if 1 <= v <= n and u not in s.trecere:
            o.append(v)
        if u in s.misiuni:
            o.append(s.misiuni[u][1])
    assert sorted(o) == list(range(1, n + 1))
    return o


def ils_split(eu: Euristica, ordine0, iteratii=100, seed=0, timp_max=np.inf, verbos=False,
              seg_max=SEG_MAX, k_cand=K_CAND):
    rng = np.random.default_rng(seed)
    sp = Split(eu, k_cand=k_cand)
    best = list(ordine0)
    m_best = sp.evalueaza(best)
    t0 = time.perf_counter()
    for it in range(iteratii):
        if time.perf_counter() - t0 > timp_max:
            break
        o = list(best)
        n = len(o)
        h = min(seg_max, n)
        tip = rng.integers(3)
        a = int(rng.integers(n))
        b = min(n, a + int(rng.integers(1, h + 1)))
        if tip == 0:
            o[a:b] = o[a:b][::-1]
        elif tip == 1:
            seg = o[a:b]
            rest = o[:a] + o[b:]
            p = int(np.clip(a + rng.integers(-h, h + 1), 0, len(rest)))
            o = rest[:p] + seg + rest[p:]
        else:
            w0 = int(rng.integers(max(1, n - 3 * h)))
            if n >= 8:
                x, y, z = sorted(rng.choice(np.arange(w0 + 1, min(n, w0 + 3 * h)), size=3,
                                            replace=False))
                o = o[:x] + o[y:z] + o[x:y] + o[z:]
        m = sp.evalueaza(o)
        if m < m_best - 1e-9:
            best, m_best = o, m
            if verbos:
                print(f"  split-ils {it:4d}: {m_best:.2f}", flush=True)
    s, m = sp.solutie(best)
    return s, m
