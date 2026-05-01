# A Validated Python Pipeline for Coastal Lagrangian Dispersal Modelling with OceanParcels 3.x and CMEMS
### Resolving Dependency Conflicts and NEMO Grid Validation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![OceanParcels 3.0.2](https://img.shields.io/badge/OceanParcels-3.0.2-green.svg)](https://oceanparcels.org)
[![CMEMS](https://img.shields.io/badge/data-CMEMS%20NEMO--OPA-blue.svg)](https://marine.copernicus.eu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source Python pipeline for Lagrangian larval dispersal modelling in the Mediterranean Sea.
The pipeline integrates OceanParcels v3.0 with daily current fields from the Copernicus Marine
Service (CMEMS) and resolves three undocumented technical problems that currently prevent
application of OceanParcels 3.x to CMEMS-forced coastal simulations.

As a case study, the pipeline models dispersal of Cyphonautes larvae of the marine bryozoan
*Electra posidoniae* from six Mediterranean sampling localities across six monthly replicates
spanning the full January–June reproductive season.

> **Repository:** https://github.com/CyberTechSea/marine-larval-dispersal

---

## Overview

| Parameter | Value |
|-----------|-------|
| Species (case study) | *Electra posidoniae* (Bryozoa) — Cyphonautes larvae |
| Pelagic Larval Duration (PLD) | 60 days (Temkin & Zimmer 2002) |
| Release sites | 6 (GE, EB, NA, TA, PA, MS) |
| Particles per site per replicate | 300 |
| Replicates | 6 (monthly: Jan–Jun) |
| Total particles | 10,800 |
| Forcing data | CMEMS MEDSEA_MULTIYEAR_PHY_006_004 |
| Temporal coverage | 1993–2002 (daily fields) |
| Advection scheme | Runge-Kutta 4th order (OceanParcels RK4) |

---

## Repository Structure

```
marine-larval-dispersal/
├── README.md                    # This file
├── environment.yml              # Pinned Conda environment for exact reproducibility
├── larval_dispersal.py          # Main simulation script
├── plot_comprehensive.py        # Per-site comprehensive figures (panels + boxplot + retention)
├── plot_overview.py             # Overview maps and connectivity figures
└── figures/                     # Example output figures
    ├── overview_all_sites.png
    ├── seasonal_patterns_overview.png
    └── connectivity_matrix.png
```

---

## Requirements

### Software
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda (recommended)
- A free account at [marine.copernicus.eu](https://marine.copernicus.eu) for data download

### Hardware

The pipeline was developed and tested on an **Apple MacBook Pro with M2 Pro chip, 16 GB RAM,
1 TB SSD**. The following minimum configurations are recommended:

| Platform | CPU | RAM | Free disk space | Notes |
|----------|-----|-----|-----------------|-------|
| **macOS** (Apple Silicon M1/M2/M3) | Apple M1 or newer | 8 GB | 35 GB | Tested on M2 Pro. Native ARM build via conda-forge |
| **macOS** (Intel) | Intel Core i5 or newer | 8 GB | 35 GB | x86_64 build |
| **Windows 10/11** | Intel/AMD quad-core | 8 GB | 35 GB | Use Anaconda Prompt or WSL2 |
| **Linux** (Ubuntu 20.04+) | Intel/AMD quad-core | 8 GB | 35 GB | Tested on Ubuntu 22.04 |
| **Virtual Machine** | 4 vCPU | 8 GB | 35 GB | Allocate at least 4 vCPU; simulation may be 3–5x slower |

**Disk space breakdown:**
- CMEMS forcing data (NetCDF): ~15–25 GB
- Trajectory output (Zarr): ~2–5 GB
- Figures and CSV outputs: ~500 MB
- Conda environment: ~3–5 GB

**Simulation runtime** (36 site x replicate combinations, 300 particles each):
- Apple M2 Pro (16 GB RAM): ~45 minutes
- Intel Core i7 (2020, 16 GB RAM): ~60–90 minutes
- Virtual machine (4 vCPU, 8 GB RAM): ~2–3 hours

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/CyberTechSea/marine-larval-dispersal.git
cd marine-larval-dispersal
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate sim_env
```

Or manually, step by step:

```bash
conda create -n sim_env python=3.11 -y
conda activate sim_env
conda install -c conda-forge parcels=3.0.2 numpy=1.26.4 zarr=2.16.1 \
  numcodecs=0.11.0 netCDF4 xarray cartopy matplotlib pandas scipy -y
pip install copernicusmarine
```

> **Warning — dependency conflict:**
> `copernicusmarine` requires `zarr >= 2.18` and `numpy >= 2.1`,
> which conflict with OceanParcels 3.0.2.
> After downloading your data, re-pin the conflicting packages:
> ```bash
> pip install "numpy==1.26.4" "zarr==2.16.1" --force-reinstall
> ```
> See [Known Issues](#known-issues-and-notes) for full details.

---

## Data Download

### 1. Login to Copernicus Marine Service

```bash
copernicusmarine login
```

Enter your credentials when prompted (one-time setup, stored locally).

### 2. Download daily current fields

```bash
copernicusmarine subset \
  --dataset-id cmems_mod_med_phy-cur_my_4.2km_P1D-m \
  --variable uo \
  --variable vo \
  --minimum-longitude 3.0 \
  --maximum-longitude 22.0 \
  --minimum-latitude 35.0 \
  --maximum-latitude 47.0 \
  --minimum-depth 1 \
  --maximum-depth 43 \
  --start-datetime 1993-01-01T00:00:00 \
  --end-datetime   2002-06-30T00:00:00 \
  --output-filename med_currents_1993_2002.nc \
  --output-directory /path/to/your/data/
```

- Expected file size: **~15–25 GB**
- Expected download time: **30–90 minutes** depending on connection speed

After download, re-pin zarr and numpy before running simulations (see above).

---

## Running the Simulation

### Setting file paths

The scripts use environment variables for file paths, with automatic fallback to the
current working directory. You have two options:

**Option A — Environment variables (recommended):**

```bash
# macOS / Linux
export NC_FILE=/path/to/med_currents_1993_2002.nc
export OUTDIR=/path/to/output/results
export SIM_YEAR=1993

# Windows (Anaconda Prompt)
set NC_FILE=C:\path\to\med_currents_1993_2002.nc
set OUTDIR=C:\path\to\output\results
set SIM_YEAR=1993
```

**Option B — Edit the scripts directly:**

Open `larval_dispersal.py` and edit the three lines at the top of the PATHS section:

```python
NC_FILE = os.environ.get("NC_FILE", "med_currents_1993_2002.nc")  # <- edit default here
OUTDIR  = os.environ.get("OUTDIR",  os.path.join(os.getcwd(), "results"))  # <- edit here
YEAR    = int(os.environ.get("SIM_YEAR", "1993"))  # <- edit year here
```

### 1. Run the main simulation

```bash
conda activate sim_env
python larval_dispersal.py
```

This will:
1. Validate grid cells for all release sites (detects NEMO land cells)
2. Build the OceanParcels FieldSet from the CMEMS NetCDF
3. Run 36 simulations (6 sites x 6 monthly replicates x 300 particles)
4. Compute dispersal distance statistics and connectivity matrix
5. Generate all figures

### 2. Re-generate comprehensive per-site figures

Reads existing Zarr trajectory files — no re-simulation needed:

```bash
# macOS / Linux
export TRAJ_DIR=/path/to/results
export OUTDIR=/path/to/results
python plot_comprehensive.py

# Windows
set TRAJ_DIR=C:\path\to\results
set OUTDIR=C:\path\to\results
python plot_comprehensive.py
```

### 3. Re-generate overview and connectivity figures

```bash
export TRAJ_DIR=/path/to/results
export OUTDIR=/path/to/results
python plot_overview.py
```

---

## Output Files

### Trajectory files (Zarr format)

One directory per site x replicate (36 total):

```
traj_GE_Jan.zarr   traj_GE_Feb.zarr   traj_GE_Mar.zarr   ...   traj_GE_Jun.zarr
traj_EB_Jan.zarr   ...
...
traj_MS_Jun.zarr
```

Each Zarr directory contains particle longitude, latitude, depth, and age at 24-hour intervals.

### Statistics (CSV)

| File | Contents |
|------|----------|
| `dispersal_statistics.csv` | Mean, median, max, P95, SD distance and % retention per site x replicate |
| `connectivity_matrix.csv` | 6x6 matrix: % particles settling within 50 km of each destination site |

### Figures (PNG, 200 dpi)

| File | Description |
|------|-------------|
| `comprehensive_<SITE>.png` | 6 monthly panels + summary map + dispersal boxplot + retention barplot |
| `site_<SITE>_6rep.png` | 6 separate monthly panels per site |
| `site_<SITE>_summary.png` | All months overlaid, coloured by release month |
| `seasonal_patterns_overview.png` | 6-panel overview: one panel per month, all sites |
| `overview_all_sites.png` | Single overview map, all sites and all months |
| `connectivity_matrix.png` | Heatmap with percentage annotations |
| `dispersal_boxplot.png` | Boxplot of dispersal distances per site per replicate |

---

## Simulation Design

### Release sites

> **Note on grid cell validation:** Before simulation, each coordinate is checked against
> the NEMO grid. Cells with fill value |u| > 10^10 m/s are land or missing-data points.
> The `find_ocean_cell()` function in `larval_dispersal.py` automatically scans the
> neighbourhood and relocates affected sites to the nearest valid ocean cell.

| Code | Locality | Longitude (E) | Latitude (N) | Sea |
|------|----------|--------------|-------------|-----|
| GE | Genova | 8.900 | 44.400 | Ligurian Sea |
| EB | Elba / Livorno | 10.200 | 43.500 | N. Tyrrhenian Sea |
| NA | Ischia / Napoli | 14.200 | 40.800 | S. Tyrrhenian Sea |
| TA | Taranto | 17.140 | 40.440 | Gulf of Taranto |
| PA | Palermo | 13.400 | 38.100 | Sicily Channel |
| MS | Messina | 15.500 | 38.200 | Strait of Messina |

### Monthly replicates

| Replicate | Release date | End date (PLD = 60 days) |
|-----------|-------------|--------------------------|
| Jan | 01 January 1993 | 01 March 1993 |
| Feb | 02 February 1993 | 03 April 1993 |
| Mar | 02 March 1993 | 01 May 1993 |
| Apr | 02 April 1993 | 01 June 1993 |
| May | 02 May 1993 | 01 July 1993 |
| Jun | 02 June 1993 | 01 August 1993 |

### Custom kernels

Four kernels are applied at each timestep (dt = 20 minutes):

- **`AdvectionRK4`** — 4th-order Runge-Kutta particle advection (OceanParcels built-in)
- **`boundary_check`** — removes particles exiting the domain (lon < 4E or > 21E; lat < 36N or > 46N), with 1-degree inner margin to prevent RK4 sub-steps from sampling outside the NetCDF grid
- **`age_kernel`** — increments particle age by dt/86400 days per timestep
- **`settle_kernel`** — removes particles when age >= PLD (60 days)

---

## Known Issues and Notes

### Dependency conflict between OceanParcels and copernicusmarine

OceanParcels 3.0.2 requires `zarr==2.16.x` and `numpy==1.26.x`.
`copernicusmarine 2.4.0` requires `zarr>=2.18` and `numpy>=2.1`.
Installing both simultaneously causes one to silently overwrite the other.

**Solution:** install `copernicusmarine`, download your data, then re-pin:

```bash
pip install "numpy==1.26.4" "zarr==2.16.1" --force-reinstall
```

### OceanParcels 3.x — recovery parameter removed

The `recovery` parameter of `ParticleSet.execute()` was removed in OceanParcels 3.x
without complete migration documentation. Out-of-bounds particles are handled by
the `boundary_check` kernel instead.

### Shapely RuntimeWarning

A `RuntimeWarning: invalid value encountered in create_collection` from Shapely is
expected and harmless. It originates from degenerate coastal geometries in the
Natural Earth dataset used by Cartopy and does not affect simulation results.

### NEMO grid cell validation

Coastal release coordinates may coincide with land cells in the NEMO grid
(fill value = 1x10^20 m/s), producing zero-velocity trajectories without any
error or warning message. Always run `find_ocean_cell()` before deploying
new release sites. See the Taranto example in the Release Sites table above.

---

## Adapting to Other Species

To apply the pipeline to a different species or region, edit the following
parameters in `larval_dispersal.py`:

| Parameter | Variable | Description |
|-----------|----------|-------------|
| Species PLD | `PLD_DAYS` | Pelagic larval duration in days |
| Release sites | `SITES` dict | lon, lat, name, color for each site |
| Particle count | `N_PARTICLES` | Particles per site per replicate |
| Release dates | `REPLICATES` list | Month and day of each replicate |
| Connectivity threshold | `SETTLE_KM` | Settlement radius in km |
| Release depth | `deps = np.full(..., 5.0)` | Depth in metres |

The forcing data can be replaced with any CMEMS product providing `uo` and `vo`
velocity components. Adjust the spatial domain and depth range in the
`copernicusmarine` download command accordingly.

---

## References

- Delandmeter P. & Van Sebille E. (2019). The Parcels v2.0 Lagrangian framework:
  new field interpolation schemes. *Geoscientific Model Development*, 12: 3571–3584.
  https://doi.org/10.5194/gmd-12-3571-2019
- Escoffier N. et al. (2021). CMEMS Mediterranean Sea Physics Reanalysis
  (MEDSEA_MULTIYEAR_PHY_006_004). Copernicus Marine Service.
  https://doi.org/10.48670/mds-00375
- Temkin I. & Zimmer R.L. (2002). Phylum Bryozoa. In: Young C.M. (ed.)
  *Atlas of Marine Invertebrate Larvae*, pp. 411–427. Academic Press, London.
- van Sebille E. et al. (2018). Lagrangian ocean analysis: fundamentals and practices.
  *Ocean Modelling*, 121: 49–75. https://doi.org/10.1016/j.ocemod.2017.11.008

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use these scripts in your research, please cite:

> [AUTHORS] (2025). A Validated Python Pipeline for Coastal Lagrangian Dispersal
> Modelling with OceanParcels 3.x and CMEMS: Resolving Dependency Conflicts and
> NEMO Grid Validation (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.[XXXXXXX]
>
> GitHub: https://github.com/CyberTechSea/marine-larval-dispersal
