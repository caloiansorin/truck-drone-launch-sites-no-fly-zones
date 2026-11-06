from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from puncte_lansare import CRS_M, _osmium, judete

RAD = Path(__file__).resolve().parent.parent
IESIRE = RAD / "data" / "zone_restrictie.geojson"
PUNCTE = RAD / "data" / "puncte_lansare.csv"

RAZA_KM = 3.0
INCLUDE_CENTRALE = False
EXCLUSE_MILITAR = {"bunker", "trench"}
EXCLUSE_NUME = ("Hotelul militar", "Barăci abandonate", "Cercul Militar", "Jandarmeria")


def _categorie(p, centrale=None):
    centrale = INCLUDE_CENTRALE if centrale is None else centrale
    if "aeroway" in p:
        return "aviatie"
    if p.get("amenity") == "prison":
        return "penitenciar"
    if p.get("power") == "plant":
        if centrale and p.get("plant:source") in ("hydro", "gas", "coal", "oil") and \
                not p.get("name", "").lower().startswith("micro"):
            return "centrala"
        return None
    if p.get("landuse") == "military" or "military" in p:
        if p.get("military") in EXCLUSE_MILITAR or p.get("name", "").startswith(EXCLUSE_NUME):
            return None
        return "militar"
    return None


def elemente_risc(centrale=None):
    filtru = ["nwr/aeroway=aerodrome,airstrip,heliport,helipad", "nwr/landuse=military",
              "nwr/military", "nwr/amenity=prison", "nwr/power=plant"]
    rand = []
    for p, g in _osmium(filtru, "risc"):
        cat = _categorie(p, centrale)
        if cat:
            rand.append(dict(categorie=cat, nume=p.get("name", ""),
                             osm=f"{p['@type']}/{p['@id']}", geometry=g))
    gdf = gpd.GeoDataFrame(rand, crs=4326).to_crs(CRS_M)
    gdf = gdf[gdf.intersects(judete())]
    gdf = gdf.assign(_wkb=gdf.geometry.normalize().to_wkb()).drop_duplicates("_wkb")
    return gdf.drop(columns="_wkb")


def zone(raza_km=RAZA_KM, centrale=None):
    E = elemente_risc(centrale)
    Z = E.assign(geometry=E.geometry.buffer(raza_km * 1000.0))
    return E, Z


class Restrictii:

    def __init__(self, raza_km=RAZA_KM, centrale=None):
        _, Z = zone(raza_km, centrale)
        self.uniune = shapely.unary_union(Z.geometry.values)
        shapely.prepare(self.uniune)
        self._crs = gpd.GeoSeries([], crs=4326).crs

    def in_zona(self, lat, lon):
        p = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(CRS_M)
        return shapely.intersects(self.uniune, p.values)

    def zbor_permis(self, lat1, lon1, lat2, lon2):
        a = gpd.GeoSeries(gpd.points_from_xy(np.atleast_1d(lon1), np.atleast_1d(lat1)),
                          crs=4326).to_crs(CRS_M).values
        b = gpd.GeoSeries(gpd.points_from_xy(np.atleast_1d(lon2), np.atleast_1d(lat2)),
                          crs=4326).to_crs(CRS_M).values
        seg = shapely.linestrings(np.stack([np.stack([shapely.get_x(a), shapely.get_y(a)], 1),
                                            np.stack([shapely.get_x(b), shapely.get_y(b)], 1)],
                                           1))
        return ~shapely.intersects(self.uniune, seg)


def main():
    E, Z = zone()
    print(f"elemente de risc: {len(E)}")
    print(E.groupby("categorie").size().to_string())
    jud = judete()
    uniune = shapely.unary_union(Z.geometry.values)
    print(f"suprafata restrictionata: {uniune.intersection(jud).area / 1e6:.0f} km2 "
          f"din {jud.area / 1e6:.0f} km2 ({uniune.intersection(jud).area / jud.area:.1%})")
    Z.to_crs(4326).to_file(IESIRE, driver="GeoJSON")

    P = pd.read_csv(PUNCTE)
    pts = gpd.GeoSeries(gpd.points_from_xy(P.lon, P.lat), crs=4326).to_crs(CRS_M)
    P["in_zona_restrictie"] = shapely.intersects(uniune, pts.values)
    P.to_csv(PUNCTE, index=False)
    print("puncte de lansare in zone:", P.groupby("sursa").in_zona_restrictie.sum().to_dict(),
          f"-> raman {(~P.in_zona_restrictie).sum()} din {len(P)}")

    L = pd.read_csv(RAD / "data" / "localitati_noduri.csv")
    lp = gpd.GeoSeries(gpd.points_from_xy(L.lon, L.lat), crs=4326).to_crs(CRS_M)
    inz = shapely.intersects(uniune, lp.values)
    print(f"localitati (clienti potentiali) in zone: {inz.sum()} din {len(L)}:",
          ", ".join(L.nume[inz].str.replace(r" Județul.*", "", regex=True).head(15)))

    from puncte_lansare import harta, zone_construite
    import retea_osm
    Pg = gpd.GeoDataFrame(P[~P.in_zona_restrictie], geometry=gpd.points_from_xy(
        P.lon[~P.in_zona_restrictie], P.lat[~P.in_zona_restrictie]), crs=4326).to_crs(CRS_M)
    harta(Pg, jud, zone_construite(), retea_osm.incarca_graf(), zone=Z)


if __name__ == "__main__":
    main()
