from pathlib import Path

import numpy as np
import networkx as nx
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

RADACINA = Path(__file__).resolve().parent.parent
DATE_VECHI = RADACINA.parent / "TSP-D-262" / "data"
DATE_NOI = RADACINA / "data"
CALE_GRAF = DATE_NOI / "bacau_neamt_drive.graphml"
CALE_LOCALITATI = DATE_NOI / "localitati_noduri.csv"
CALE_VALIDARE = DATE_NOI / "validare_distante.csv"

CALE_OSM_FILTRAT = DATE_NOI / "osm" / "bn_drive.osm"

PREGATIRE_OSM = """
curl -L -o data/osm/romania-latest.osm.pbf https://download.geofabrik.de/europe/romania-latest.osm.pbf
# poligonul celor doua judete, cu buffer de 15 km (drumuri care ies scurt din judet)
python -c "import osmnx as ox; g=ox.geocode_to_gdf(['R2248681','R2248675'], by_osmid=True); \\
  g=g[['geometry']].dissolve().to_crs(3844); g['geometry']=g.buffer(15000).simplify(200); \\
  g.to_crs(4326).to_file('data/osm/bacau_neamt_buf15km.geojson', driver='GeoJSON')"
osmium extract -p data/osm/bacau_neamt_buf15km.geojson data/osm/romania-latest.osm.pbf -o data/osm/bn.osm.pbf
osmium tags-filter data/osm/bn.osm.pbf w/highway=motorway,motorway_link,trunk,trunk_link,\\
  primary,primary_link,secondary,secondary_link,tertiary,tertiary_link,unclassified,\\
  residential,living_street,road -o data/osm/bn_drive.osm
"""


def incarca_graf(reconstruieste=False):
    gz = CALE_GRAF.with_name(CALE_GRAF.name + '.gz')
    if not CALE_GRAF.exists() and gz.exists() and not reconstruieste:
        import gzip
        import shutil
        with gzip.open(gz, 'rb') as fi, open(CALE_GRAF, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
    if CALE_GRAF.exists() and not reconstruieste:
        return ox.load_graphml(CALE_GRAF)
    if not CALE_OSM_FILTRAT.exists():
        raise FileNotFoundError(f"lipseste {CALE_OSM_FILTRAT}; see README.md")
    G = ox.graph_from_xml(CALE_OSM_FILTRAT, bidirectional=False, simplify=True, retain_all=False)
    G = ox.truncate.largest_component(G, strongly=True)
    ox.save_graphml(G, CALE_GRAF)
    return G


def citeste_localitati():
    df = pd.read_excel(DATE_VECHI / "Orase.xlsx", header=None,
                       names=["id", "lat", "lon", "nume"])
    return df.set_index("id")


def citeste_matrice_veche():
    df = pd.read_excel(DATE_VECHI / "matrice_distante.xlsx", index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df


def matrice_adiacenta(G):
    noduri = list(G.nodes)
    poz = {v: i for i, v in enumerate(noduri)}
    lung = {}
    for u, v, d in G.edges(data=True):
        cheie = (poz[u], poz[v])
        lung[cheie] = min(lung.get(cheie, np.inf), d["length"])
    rand, col = zip(*lung)
    A = csr_matrix((list(lung.values()), (rand, col)), shape=(len(noduri), len(noduri)))
    return A, noduri, poz


def ataseaza_localitati(G, loc):
    noduri = ox.distance.nearest_nodes(G, X=loc["lon"].to_numpy(), Y=loc["lat"].to_numpy())
    rez = loc.copy()
    rez["nod_osm"] = noduri
    rez["dist_atasare_m"] = [
        ox.distance.great_circle(la, lo, G.nodes[n]["y"], G.nodes[n]["x"])
        for la, lo, n in zip(loc["lat"], loc["lon"], noduri)
    ]
    return rez


def distante_intre_localitati(A, poz, loc):
    surse = [poz[n] for n in loc["nod_osm"]]
    D = dijkstra(A, directed=True, indices=surse)
    return D[:, surse]


def main():
    G = incarca_graf()
    print(f"graf OSM: {G.number_of_nodes()} noduri, {G.number_of_edges()} arce")

    loc = ataseaza_localitati(G, citeste_localitati())
    loc.to_csv(CALE_LOCALITATI)
    print("distanta de atasare (m):", loc["dist_atasare_m"].describe().round(0).to_dict())

    A, _, poz = matrice_adiacenta(G)
    D_osm = distante_intre_localitati(A, poz, loc)
    ids = list(loc.index)
    D_vechi = citeste_matrice_veche().loc[ids, ids].to_numpy(float)

    i, j = np.where(~np.eye(len(ids), dtype=bool))
    raport = D_osm[i, j] / D_vechi[i, j]
    val = pd.DataFrame({"de_la": np.array(ids)[i], "la": np.array(ids)[j],
                        "osm_m": D_osm[i, j], "vechi_m": D_vechi[i, j], "raport": raport})
    val.to_csv(CALE_VALIDARE, index=False)
    print("raport OSM / matrice veche:",
          val["raport"].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_dict())
    print("perechi cu abatere > 10%:", int((abs(raport - 1) > 0.10).sum()), "din", len(raport))


if __name__ == "__main__":
    main()
