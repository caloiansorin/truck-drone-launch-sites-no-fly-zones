from dataclasses import dataclass, replace
from itertools import product

V_REF_KMH = 50.0

GRILA = dict(
    SL_km=(0.0, 0.5, 1.0, 1.5),
    alpha=(2.0, 3.0, 4.0),
    autonomie_km=(10.0, 20.0, 30.0),
)


@dataclass(frozen=True)
class Operatiuni:
    alpha: float = 3.0
    autonomie_km: float = 20.0
    SL_km: float = 1.0
    Q_max_kg: float = 3.0
    cerere_max_kg: float = 5.0
    t_servire_min: float = 0.0
    v_ref_kmh: float = V_REF_KMH

    @property
    def v_drona_kmh(self):
        return self.alpha * self.v_ref_kmh

    @property
    def SL_min(self):
        return self.SL_km / self.v_ref_kmh * 60.0

    @property
    def SR_min(self):
        return self.SL_min

    def cu(self, **m):
        return replace(self, **m)

    @property
    def eticheta(self):
        return f"a{self.alpha:g}-AD{self.autonomie_km:g}-SL{self.SL_km:g}"


BAZA_REALIST = Operatiuni(alpha=70.0 / V_REF_KMH, autonomie_km=20.0, SL_km=V_REF_KMH / 60.0)
SENSIBILITATE = dict(
    v_drona_kmh=(50.0, 70.0, 90.0),
    SL_min=(1.0, 2.0, 3.0),
    autonomie_km=(10.0, 20.0, 30.0),
)


def configuratii():
    return [Operatiuni(alpha=a, autonomie_km=ad, SL_km=sl)
            for sl, a, ad in product(GRILA["SL_km"], GRILA["alpha"], GRILA["autonomie_km"])]


if __name__ == "__main__":
    for o in configuratii()[:3] + configuratii()[-1:]:
        print(o.eticheta, f"v_drona {o.v_drona_kmh:g} km/h, SL=SR {o.SL_min:.1f} min")
    print(len(configuratii()), "configuratii MDPI")
    b = BAZA_REALIST
    print(f"baza realista: v_drona {b.v_drona_kmh:g} km/h, SL=SR {b.SL_min:.1f} min "
          f"({b.SL_km:.2f} km-echiv.), autonomie {b.autonomie_km:g} km, Q_max {b.Q_max_kg:g} kg")
