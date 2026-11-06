# Truck–drone delivery with rule-based launch sites and no-fly zones

Code, derived data and results for the paper

> S.-I. Caloian, *Truck–Drone Delivery with Rule-Based Launch Sites and No-Fly Zones on a Real Road Network* (submitted to BRAIN. Broad Research in Artificial Intelligence and Neuroscience).

The study covers Bacău and Neamț counties (Romania): 175 customer localities, one depot in Bacău, a directed OpenStreetMap road graph with 50,013 nodes, 1318 candidate launch/recovery sites (parking lots, rest areas, road shoulders) and 59 no-fly zones (3 km around aviation, military and prison sites mapped in OpenStreetMap).

## Contents

| Path | Content |
|---|---|
| `src/` | All code used for the paper (Python 3.12). |
| `data/customers.csv` | Customer localities, coordinates and parcel masses (depot first; masses drawn U[0, 5] kg, seed 42). |
| `data/localitati_noduri.csv` | Road-graph node attached to each locality. |
| `data/puncte_lansare.csv` | The 1318 candidate launch/recovery sites with OSM id, type (`parcare` = parking lot, `refugiu` = rest area / services, `acostament` = road shoulder), attachment node, road class, flags for built-up area and no-fly zone. |
| `data/risk_sites.csv`, `data/zone_restrictie.geojson` | The 59 risk sites (category, name, OSM id) and the 3 km no-fly zones. |
| `data/bacau_neamt_drive.graphml.gz` | The routable road graph (largest strongly connected component). It is decompressed automatically on first use. |
| `data/osm/` | County boundary and the 15 km buffer used to cut the OSM extract. |
| `data/validare_distante.csv` | Comparison of road-graph travel distances with the distances of the earlier study. |
| `results/` | All result tables used in the paper (CSV) and the figures (`results/figuri/`). |

## Data source and date

Road network, parking lots, rest areas, built-up areas and risk sites come from OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0), Geofabrik extract for Romania downloaded on 2 November 2026; the latest object timestamp in the extract is 2026-11-01T21:19:22Z. The derived files in `data/` are distributed under the same ODbL licence.

To rebuild the derived data from scratch, download the extract and cut it with `osmium` (the exact commands are in the string `PREGATIRE_OSM` in `src/retea_osm.py`), then run `src/puncte_lansare.py` and `src/zone_restrictie.py`.

## Installation

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`osmium-tool` is needed only to rebuild the data from the raw OSM extract.

## Reproducing the results

Run from the repository root. Instance matrices are cached in `cache/` on first use.

| Paper item | Command | Output |
|---|---|---|
| Table 5, Figure 2 (comparison with the exact optimum, 30 instances) | `.venv/bin/python src/etalon_dp.py 8 10` | `results/etalon_dp_n8.csv` |
| Table 6, Figure 3 (main experiment, 10 ILS runs) | `OMP_NUM_THREADS=1 .venv/bin/python src/revizie.py rulari 7` | `results/revizie_rulari.csv` |
| Table 7 (ten parcel-mass draws) | `OMP_NUM_THREADS=1 .venv/bin/python src/revizie.py mase 7` | `results/revizie_mase.csv` |
| Table 8 (candidate-list size K) | `OMP_NUM_THREADS=1 .venv/bin/python src/revizie.py K 7` | `results/revizie_K.csv` |
| Table 9, Figure 5 (sensitivity) | `OMP_NUM_THREADS=1 .venv/bin/python src/rulari.py sensibilitate 6` | `results/sensibilitate.csv` |
| Table 10, Figure 6 (36-configuration grid) | `OMP_NUM_THREADS=1 .venv/bin/python src/rulari.py mdpi 6` | `results/grila_mdpi.csv` |
| Re-runs of the grid cases where a larger site set did worse | `OMP_NUM_THREADS=1 .venv/bin/python src/revizie.py inversiuni 7` | `results/revizie_inversiuni.csv` |
| Figures | `.venv/bin/python src/figuri.py toate` | `results/figuri/` |

The last argument is the number of parallel processes.

### Random seeds

- Parcel masses: seed 42 for the main instance (`data/customers.csv`); seeds 1–10 for the robustness test (`numpy.random.default_rng(seed).uniform(0, 5)`, rounded to 0.01 kg).
- Truck tour: nearest neighbour + 2-opt/Or-opt with 20 randomized restarts, `numpy.random.default_rng(0)`.
- Iterated local search: `numpy.random.default_rng(seed)` with the seed given in the `seed` column of each results file (0–9 for the main experiment, 0–2 for the other experiments, 0–1 for the sensitivity analysis, 0 for the 36-configuration grid).
- Small instances: groups of 9 neighbouring localities chosen with seeds 0–9; the ILS on instance `s` uses seed `s`.

Every solution reported in the results files was checked by the independent feasibility routine `Euristica.verifica` (payload, range including hovering, admissible sites, no-fly zones, non-overlapping sorties, each customer served once).

## Main modules

| Module | Role |
|---|---|
| `retea_osm.py`, `retea_realista.py` | Road graph and class-dependent truck speeds. |
| `puncte_lansare.py` | Candidate sites: parking lots, rest areas and road shoulders. |
| `zone_restrictie.py` | Risk sites, 3 km no-fly zones, check of straight drone legs. |
| `instanta_realista.py` | Instance: travel-time and flight-distance matrices, site sets C, P, S. |
| `parametri_operationali.py` | Drone parameters (baseline, sensitivity, 36-configuration grid). |
| `tur_camion.py` | Truck-only tour. |
| `euristica.py` | Solution model, evaluation, feasibility check, greedy insertion. |
| `split_dp.py` | Split dynamic program with roadside sites and the iterated local search. |
| `dp_exact.py`, `etalon_dp.py` | Exact subset dynamic program and the comparison on small instances. |
| `experiment.py`, `rulari.py`, `revizie.py` | Experiments. |
| `figuri.py` | Figures. |

Identifiers in the code are in Romanian (`clienti` = customers, `parcari` = parking lots and rest areas, `toate` = all sites including road shoulders, `misiune` = drone sortie, `camion` = truck).
