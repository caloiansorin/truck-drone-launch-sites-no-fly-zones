import json
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

import retea_osm

RAD = Path(__file__).resolve().parent.parent
OSM = RAD / "data" / "osm"
PBF = OSM / "bn.osm.pbf"
IESIRE = RAD / "data" / "puncte_lansare.csv"
HARTA = RAD / "results" / "harta_puncte_lansare.png"

CRS_M = 3844
DIST_ATASARE_MAX = 300.0
BUFFER_CONSTRUIT_M = 100.0
RAZA_CENTRU_LOC_M = 500.0
DIST_INTERSECTIE_M = 100.0
LUNGIME_ARC_MIN = 2 * DIST_INTERSECTIE_M + 100.0
PAS_RURAL_M = 2000.0
GRILA_RURAL_M = 2000.0
CLASE_RURAL = {"secondary", "tertiary", "unclassified"}
AEROPORT_BACAU = (46.5219, 26.9103)
RAZA_AEROPORT_KM = 5.0
EXCLUS_ACCESS = {"private", "no", "permit"}
EXCLUS_PARKING = {"garage", "garage_boxes", "multi-storey", "underground", "carports"}


def _osmium(filtru, nume, tipuri=None):
    pbf, gj = OSM / f"{nume}.osm.pbf", OSM / f"{nume}.geojsonseq"
    if not gj.exists():
        subprocess.run(["osmium", "tags-filter", PBF, *filtru, "-o", pbf, "--overwrite"],
                       check=True)
        cmd = ["osmium", "export", pbf, "-f", "geojsonseq", "-o", gj, "--overwrite",
               "-a", "type,id"]
        if tipuri:
            cmd += ["--geometry-types", tipuri]
        subprocess.run(cmd, check=True)
    for linie in open(gj):
        f = json.loads(linie.strip().lstrip("\x1e"))
        yield f["properties"], shapely.geometry.shape(f["geometry"])


def judete():
    return gpd.read_file(OSM / "bacau_neamt.geojson").to_crs(CRS_M).union_all()


def zone_construite():
    geom = [g for _, g in _osmium(["wr/landuse=residential"], "rezidential", "polygon")]
    z = gpd.GeoSeries(geom, crs=4326).to_crs(CRS_M).buffer(0).buffer(BUFFER_CONSTRUIT_M)
    loc = [g for _, g in _osmium(["n/place=city,town,village,hamlet"], "localitati")]
    c = gpd.GeoSeries(loc, crs=4326).to_crs(CRS_M).buffer(RAZA_CENTRU_LOC_M)
    return shapely.unary_union(np.concatenate([z.values, c.values]))


def parcari(jud):
    rand = []
    for p, g in _osmium(["nwr/amenity=parking", "nwr/highway=rest_area,services"], "parcari"):
        if p.get("access") in EXCLUS_ACCESS or p.get("parking") in EXCLUS_PARKING:
            continue
        if p.get("hgv") == "no" and p.get("amenity") == "parking":
            continue
        rand.append(dict(sursa="parcare" if p.get("amenity") == "parking" else "refugiu",
                         osm=f"{p['@type']}/{p['@id']}", nume=p.get("name", ""),
                         geometry=g))
    gdf = gpd.GeoDataFrame(rand, crs=4326).to_crs(CRS_M)
    gdf["arie_m2"] = gdf.geometry.area
    gdf["geometry"] = gdf.geometry.representative_point()
    return gdf[gdf.within(jud)].drop_duplicates("osm")


def noduri_graf(G):
    noduri = list(G.nodes)
    pts = gpd.GeoSeries(gpd.points_from_xy([G.nodes[v]["x"] for v in noduri],
                                           [G.nodes[v]["y"] for v in noduri]), crs=4326)
    xy = np.column_stack([pts.to_crs(CRS_M).x, pts.to_crs(CRS_M).y])
    return noduri, xy


def ataseaza(gdf, noduri, xy):
    d, i = cKDTree(xy).query(np.column_stack([gdf.geometry.x, gdf.geometry.y]))
    gdf = gdf.assign(nod_osm=np.asarray(noduri)[i], dist_atasare_m=d)
    gdf = gdf[gdf.dist_atasare_m <= DIST_ATASARE_MAX]
    return gdf.sort_values("arie_m2", ascending=False).drop_duplicates("nod_osm")


def _clasa(h):
    h = h if isinstance(h, list) else [h]
    return next((x for x in h if x in CLASE_RURAL), None)


def acostamente(G, jud, construit):
    vazut, rand = set(), []
    for u, v, d in G.edges(data=True):
        cls = _clasa(d.get("highway"))
        if cls is None or d.get("bridge") or d.get("tunnel") or d["length"] < LUNGIME_ARC_MIN:
            continue
        cheie = (min(u, v), max(u, v), str(d.get("osmid")))
        if cheie in vazut:
            continue
        vazut.add(cheie)
        geom = d.get("geometry") or shapely.LineString(
            [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        L = d["length"]
        k = max(1, int(L // PAS_RURAL_M))
        for s in L * np.arange(1, k + 1) / (k + 1):
            rand.append(dict(sursa="acostament", osm=f"way/{d.get('osmid')}", nume=d.get("name", ""),
                             clasa_drum=cls, u=u, v=v, offset_m=s, lungime_arc_m=L,
                             dublu_sens=G.has_edge(v, u),
                             geometry=geom.interpolate(s / L, normalized=True)))
    gdf = gpd.GeoDataFrame(rand, crs=4326).to_crs(CRS_M)
    if not len(gdf):
        return gdf
    gdf = gdf[gdf.within(jud) & ~gdf.within(construit)]
    gdf = gdf.assign(celula=list(zip((gdf.geometry.x // GRILA_RURAL_M).astype(int),
                                     (gdf.geometry.y // GRILA_RURAL_M).astype(int))))
    ordine = gdf.clasa_drum.map({"secondary": 0, "tertiary": 1, "unclassified": 2})
    gdf = gdf.assign(_o=ordine).sort_values("_o").drop_duplicates("celula")
    return gdf.drop(columns=["celula", "_o"])


def harta(puncte, jud, construit, G, zone=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    linii = [d["geometry"] for u, v, d in G.edges(data=True)
             if "geometry" in d and (_clasa(d.get("highway")) or
                                     d.get("highway") in ("primary", "trunk"))]
    fig, ax = plt.subplots(figsize=(11, 13))
    gpd.GeoSeries([jud], crs=CRS_M).boundary.plot(ax=ax, color="k", lw=0.8)
    gpd.GeoSeries([construit], crs=CRS_M).intersection(jud).plot(ax=ax, color="#f2d7c9", lw=0)
    gpd.GeoSeries(linii, crs=4326).to_crs(CRS_M).plot(ax=ax, color="#bbbbbb", lw=0.3)
    stil = {"parcare": ("#1f6fb4", 6), "refugiu": ("#2a9d3f", 14), "acostament": ("#c0392b", 6)}
    for s, (c, m) in stil.items():
        sub = puncte[puncte.sursa == s]
        sub.plot(ax=ax, color=c, markersize=m, label=f"{s} ({len(sub)})")
    if zone is None:
        aer = gpd.GeoSeries(gpd.points_from_xy([AEROPORT_BACAU[1]], [AEROPORT_BACAU[0]]), crs=4326)
        aer.to_crs(CRS_M).buffer(RAZA_AEROPORT_KM * 1000).boundary.plot(ax=ax, color="purple",
                                                                      ls="--", lw=1)
    else:
        gpd.GeoSeries([shapely.unary_union(zone.geometry.values)], crs=CRS_M).plot(
            ax=ax, color="purple", alpha=0.25, lw=0)
        gpd.GeoSeries([shapely.unary_union(zone.geometry.values)], crs=CRS_M).boundary.plot(
            ax=ax, color="purple", lw=0.8)
        ax.plot([], [], color="purple", lw=6, alpha=0.4, label=f"zone de restrictie ({len(zone)})")
    ax.set_title("Puncte admise de lansare/recuperare - Bacau si Neamt")
    ax.legend(loc="lower left")
    ax.set_axis_off()
    fig.savefig(HARTA, dpi=170, bbox_inches="tight")


def acoperire(puncte, nume, raze=(1000, 3000, 5000)):
    L = pd.read_csv(RAD / "data" / "localitati_noduri.csv")
    Lg = gpd.GeoSeries(gpd.points_from_xy(L.lon, L.lat), crs=4326).to_crs(CRS_M)
    d, _ = cKDTree(np.column_stack([puncte.geometry.x, puncte.geometry.y])).query(
        np.column_stack([Lg.x, Lg.y]))
    print(f"  {nume}: localitati ({len(L)}) cu punct la <= " +
          ", ".join(f"{r / 1000:g} km: {(d <= r).sum()}" for r in raze) +
          f"; mediana {np.median(d) / 1000:.2f} km")


def main():
    G = retea_osm.incarca_graf()
    jud, construit = judete(), zone_construite()
    noduri, xy = noduri_graf(G)

    P = parcari(jud)
    n0 = len(P)
    P = ataseaza(P, noduri, xy)
    print(f"parcari/refugii in judete: {n0}; dupa atasare (<= {DIST_ATASARE_MAX:g} m) "
          f"si un punct per nod: {len(P)}")

    A = acostamente(G, jud, construit)
    _, iA = cKDTree(xy).query(np.column_stack([A.geometry.x, A.geometry.y]))
    A = A.assign(nod_osm=np.asarray(noduri)[iA], dist_atasare_m=0.0, arie_m2=0.0)
    print(f"acostamente candidate (in afara zonelor construite, grila {GRILA_RURAL_M:g} m): {len(A)}")

    puncte = gpd.GeoDataFrame(pd.concat([P, A], ignore_index=True), crs=CRS_M)
    puncte["in_zona_construita"] = puncte.within(construit)
    ll = puncte.to_crs(4326)
    puncte["lat"], puncte["lon"] = ll.geometry.y, ll.geometry.x
    from instante_retea import haversine_km
    puncte["langa_aeroport"] = haversine_km(*AEROPORT_BACAU, puncte.lat, puncte.lon) < RAZA_AEROPORT_KM
    puncte.insert(0, "id", range(len(puncte)))

    print(puncte.groupby("sursa").agg(n=("id", "size"), construit=("in_zona_construita", "sum"),
                                      aeroport=("langa_aeroport", "sum")))
    acoperire(P, "doar parcari/refugii")
    acoperire(puncte, "cu acostamente")

    col = ["id", "sursa", "osm", "nume", "lat", "lon", "nod_osm", "dist_atasare_m", "arie_m2",
           "clasa_drum", "u", "v", "offset_m", "lungime_arc_m", "dublu_sens",
           "in_zona_construita", "langa_aeroport"]
    puncte[col].to_csv(IESIRE, index=False)
    harta(puncte, jud, construit, G)
    print(f"scris {IESIRE.name} ({len(puncte)} puncte) si {HARTA.name}")


if __name__ == "__main__":
    main()
