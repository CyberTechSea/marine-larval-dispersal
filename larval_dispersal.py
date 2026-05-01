import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import netCDF4 as nc4
import xarray as xr
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, Variable, StatusCode

# ─── PATHS ───────────────────────────────────────────────────────────────────
# Edit NC_FILE and OUTDIR before running, or set them as environment variables:
#   export NC_FILE=/path/to/med_currents_1993_2002.nc
#   export OUTDIR=/path/to/output/directory
#   export SIM_YEAR=1993
NC_FILE = os.environ.get("NC_FILE", "med_currents_1993_2002.nc")
OUTDIR  = os.environ.get("OUTDIR",  os.path.join(os.getcwd(), "results"))
YEAR    = int(os.environ.get("SIM_YEAR", "1993"))
os.makedirs(OUTDIR, exist_ok=True)

# ─── SITES ───────────────────────────────────────────────────────────────────
SITES = {
    "GE": {"lon":  8.90, "lat": 44.40, "name": "Genova",        "color": "#e53935"},
    "EB": {"lon": 10.20, "lat": 43.50, "name": "Elba/Livorno",  "color": "#fb8c00"},
    "NA": {"lon": 13.96, "lat": 40.73, "name": "Ischia/Napoli", "color": "#8e24aa"},
    "TA": {"lon": 17.14, "lat": 40.44, "name": "Taranto",       "color": "#1e88e5"},
    "PA": {"lon": 13.40, "lat": 38.10, "name": "Palermo",       "color": "#dfeb07"},
    "MS": {"lon": 15.50, "lat": 38.20, "name": "Messina",       "color": "#43a047"},
}

# ─── 6 REPLICATES — full reproductive season Jan–Jun ─────────────────────────
# Each replicate: release on day 1, track for 60 days (PLD)
# Jan: 01/01 – 01/03  (releases 01 Jan)
# Feb: 02/02 – 02/04  (releases 02 Feb)
# Mar: 02/03 – 01/05  (releases 02 Mar)
# Apr: 02/04 – 01/06  (releases 02 Apr)
# May: 02/05 – 01/07  (releases 02 May)
# Jun: 02/06 – 01/08  (releases 02 Jun)
REPLICATES = [
    {"name": "Jan", "release": (1,  1),  "color": "#1565c0"},
    {"name": "Feb", "release": (2,  2),  "color": "#0288d1"},
    {"name": "Mar", "release": (3,  2),  "color": "#00897b"},
    {"name": "Apr", "release": (4,  2),  "color": "#558b2f"},
    {"name": "May", "release": (5,  2),  "color": "#f9a825"},
    {"name": "Jun", "release": (6,  2),  "color": "#e53935"},
]

N_PARTICLES = 300
PLD_DAYS    = 60
SETTLE_KM   = 50.0


# ─── UTILITIES ───────────────────────────────────────────────────────────────
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(np.radians(lat2 - lat1) / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─── FIELDSET ────────────────────────────────────────────────────────────────
print("\n[1] Building FieldSet from: " + NC_FILE)
ds = nc4.Dataset(NC_FILE, "r")
avail = list(ds.variables.keys())
print("    Variables: " + str(avail))
u_name  = next((v for v in ["uo", "uo_avg", "vozocrtx"] if v in avail), None)
v_name  = next((v for v in ["vo", "vo_avg", "vomecrty"] if v in avail), None)
lo_name = next((v for v in ["longitude", "lon"] if v in avail), None)
la_name = next((v for v in ["latitude",  "lat"] if v in avail), None)
has_dep = "depth" in ds.variables
t_units = ds.variables["time"].units
t_cal   = getattr(ds.variables["time"], "calendar", "standard")
dates   = nc4.num2date(ds.variables["time"][:], units=t_units, calendar=t_cal)
ds.close()
print("    U=" + str(u_name) + " | " + str(dates[0]) + " -> " + str(dates[-1]))

variables = {"U": u_name, "V": v_name}
dims_u = {"lon": lo_name, "lat": la_name, "time": "time"}
dims_v = {"lon": lo_name, "lat": la_name, "time": "time"}
if has_dep:
    dims_u["depth"] = "depth"
    dims_v["depth"] = "depth"
dimensions = {"U": dims_u, "V": dims_v}

fieldset = FieldSet.from_netcdf(NC_FILE, variables=variables,
                                 dimensions=dimensions,
                                 allow_time_extrapolation=True)
print("    FieldSet OK")


# ─── PARTICLE CLASS & KERNELS ─────────────────────────────────────────────────
class LarvaParticle(JITParticle):
    age_days  = Variable("age_days",  initial=0.0, dtype=np.float32)
    start_lon = Variable("start_lon", to_write=True, dtype=np.float32)
    start_lat = Variable("start_lat", to_write=True, dtype=np.float32)


def age_kernel(particle, fieldset, time):
    particle.age_days += particle.dt / 86400.0


def boundary_check(particle, fieldset, time):
    if particle.lon < 4.0 or particle.lon > 21.0:
        particle.delete()
    if particle.lat < 36.0 or particle.lat > 46.0:
        particle.delete()


def settle_kernel(particle, fieldset, time):
    if particle.age_days >= 60.0:
        particle.delete()


# ─── SIMULATIONS ─────────────────────────────────────────────────────────────
print("\n[2] Running simulations (" + str(N_PARTICLES) + " particles × " +
      str(len(SITES)) + " sites × " + str(len(REPLICATES)) + " replicates = " +
      str(N_PARTICLES * len(SITES) * len(REPLICATES)) + " total)...")

traj_files = {}
total = len(SITES) * len(REPLICATES)
count = 0

for site_code, site in SITES.items():
    for rep in REPLICATES:
        count += 1
        print("\n  [" + str(count) + "/" + str(total) + "] " + site_code + " | " + rep["name"])
        np.random.seed(hash(site_code + rep["name"]) % 2**31)
        lons  = site["lon"] + np.random.uniform(-0.1, 0.1, N_PARTICLES)
        lats  = site["lat"] + np.random.uniform(-0.1, 0.1, N_PARTICLES)
        deps  = np.full(N_PARTICLES, 5.0)
        m, d  = rep["release"]
        rdate = datetime(YEAR, m, d)
        times = np.array([rdate] * N_PARTICLES)

        pset = ParticleSet(
            fieldset=fieldset, pclass=LarvaParticle,
            lon=lons, lat=lats, depth=deps, time=times,
            start_lon=lons, start_lat=lats,
        )

        fname = os.path.join(OUTDIR, "traj_" + site_code + "_" + rep["name"] + ".zarr")
        output  = pset.ParticleFile(name=fname, outputdt=timedelta(hours=24))
        kernels = AdvectionRK4 + pset.Kernel(boundary_check) + pset.Kernel(age_kernel) + pset.Kernel(settle_kernel)

        pset.execute(kernels, runtime=timedelta(days=PLD_DAYS),
                     dt=timedelta(minutes=20), output_file=output)

        traj_files[(site_code, rep["name"])] = fname
        print("    Saved: " + fname)


# ─── STATISTICS ──────────────────────────────────────────────────────────────
print("\n[3] Computing statistics...")
records = []
for (sc, rn), fpath in traj_files.items():
    site = SITES[sc]
    ds2  = xr.open_zarr(fpath)
    lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
    ds2.close()
    flons, flats = [], []
    for i in range(lons2.shape[0]):
        v = ~np.isnan(lons2[i])
        flons.append(lons2[i][v][-1] if v.any() else site["lon"])
        flats.append(lats2[i][v][-1] if v.any() else site["lat"])
    flons = np.array(flons); flats = np.array(flats)
    dists = haversine_km(site["lon"], site["lat"], flons, flats)
    records.append({
        "site": sc, "replicate": rn,
        "dist_mean_km":   float(np.nanmean(dists)),
        "dist_median_km": float(np.nanmedian(dists)),
        "dist_max_km":    float(np.nanmax(dists)),
        "dist_p95_km":    float(np.nanpercentile(dists, 95)),
        "dist_std_km":    float(np.nanstd(dists)),
        "pct_retained":   float(np.mean(dists < SETTLE_KM) * 100),
    })
    print("    " + sc + " " + rn + ": mean=" + str(round(np.nanmean(dists), 1)) +
          " km  max=" + str(round(np.nanmax(dists), 1)) + " km  retained=" +
          str(round(np.mean(dists < SETTLE_KM) * 100, 1)) + "%")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTDIR, "dispersal_statistics.csv"), index=False, float_format="%.2f")


# ─── CONNECTIVITY ─────────────────────────────────────────────────────────────
site_codes = list(SITES.keys())
n    = len(site_codes)
conn = np.zeros((n, n))
counts = np.zeros(n)

for (sc, rn), fpath in traj_files.items():
    s_idx = site_codes.index(sc)
    site  = SITES[sc]
    ds2   = xr.open_zarr(fpath)
    lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
    ds2.close()
    flons, flats = [], []
    for i in range(lons2.shape[0]):
        v = ~np.isnan(lons2[i])
        flons.append(lons2[i][v][-1] if v.any() else site["lon"])
        flats.append(lats2[i][v][-1] if v.any() else site["lat"])
    flons = np.array(flons); flats = np.array(flats)
    counts[s_idx] += len(flons)
    for d_idx, dc in enumerate(site_codes):
        dst = SITES[dc]
        conn[s_idx, d_idx] += np.sum(haversine_km(dst["lon"], dst["lat"], flons, flats) <= SETTLE_KM)

for i in range(n):
    if counts[i] > 0:
        conn[i, :] = conn[i, :] / counts[i] * 100

df_conn = pd.DataFrame(conn, index=site_codes, columns=site_codes)
df_conn.to_csv(os.path.join(OUTDIR, "connectivity_matrix.csv"), float_format="%.2f")
print("\nConnectivity matrix (%):")
print(df_conn.round(1).to_string())


# ─── PLOTS ────────────────────────────────────────────────────────────────────
print("\n[4] Generating plots...")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("    Cartopy OK")
except ImportError:
    HAS_CARTOPY = False
    print("    Cartopy not found — basic map")

EXTENT = [5.5, 20.0, 36.5, 45.5]


def setup_map(ax):
    if HAS_CARTOPY:
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,      facecolor="#c8b89a", zorder=2)
        ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#3a2a10", zorder=4)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#999999", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([6, 9, 12, 15, 18])
        gl.ylocator = mticker.FixedLocator([37, 39, 41, 43, 45])
        gl.xlabel_style = {"size": 6, "color": "gray"}
        gl.ylabel_style = {"size": 6, "color": "gray"}
    else:
        ax.set_facecolor("#d0e8f5")
        ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_xticks([6, 9, 12, 15, 18]); ax.set_yticks([37, 39, 41, 43, 45])
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.3, linestyle="--")


def draw_traj(ax, lons2, lats2, color, alpha=0.6, lw=0.8):
    for i in range(lons2.shape[0]):
        v = ~np.isnan(lons2[i])
        if v.sum() > 1:
            kw = dict(color=color, alpha=alpha, linewidth=lw, zorder=5)
            if HAS_CARTOPY:
                ax.plot(lons2[i][v], lats2[i][v], transform=ccrs.PlateCarree(), **kw)
            else:
                ax.plot(lons2[i][v], lats2[i][v], **kw)


def draw_marker(ax, lon, lat, color, label, ms=9, offset=(0.25, 0.18)):
    kw_p = dict(color=color, ms=ms, markeredgecolor="white", markeredgewidth=1.3, zorder=8)
    kw_t = dict(fontsize=9, fontweight="bold", color=color, zorder=9)
    if HAS_CARTOPY:
        ax.plot(lon, lat, "o", transform=ccrs.PlateCarree(), **kw_p)
        ax.text(lon + offset[0], lat + offset[1], label,
                transform=ccrs.PlateCarree(), **kw_t)
    else:
        ax.plot(lon, lat, "o", **kw_p)
        ax.text(lon + offset[0], lat + offset[1], label, **kw_t)


# --- A: Per site — 6 panels (one per replicate) ---
print("\n    [A] Per-site maps (6 replicates)...")
for sc, site in SITES.items():
    print("        " + sc + "...")
    if HAS_CARTOPY:
        fig, axes = plt.subplots(1, 6, figsize=(26, 5),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(1, 6, figsize=(26, 5))

    for r_idx, rep in enumerate(REPLICATES):
        ax = axes[r_idx]
        setup_map(ax)
        m, d = rep["release"]
        ax.set_title(rep["name"] + "\n(" + str(d).zfill(2) + "/" + str(m).zfill(2) + ")",
                     fontsize=9, fontweight="bold", color=rep["color"], pad=3)

        fpath = traj_files.get((sc, rep["name"]))
        if fpath:
            ds2   = xr.open_zarr(fpath)
            lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
            ds2.close()
            draw_traj(ax, lons2, lats2, site["color"], alpha=0.6, lw=0.8)

        for other_sc, other_site in SITES.items():
            if other_sc == sc:
                continue
            kw = dict(color=other_site["color"], ms=4, markeredgecolor="white",
                      markeredgewidth=0.7, alpha=0.45, zorder=7)
            if HAS_CARTOPY:
                ax.plot(other_site["lon"], other_site["lat"], "o",
                        transform=ccrs.PlateCarree(), **kw)
            else:
                ax.plot(other_site["lon"], other_site["lat"], "o", **kw)

        draw_marker(ax, site["lon"], site["lat"], site["color"], sc, ms=10)

    fig.suptitle(
        sc + " — " + site["name"] + "   |   Electra posidoniae Cyphonautes\n"
        "CMEMS NEMO-OPA 4.2 km  ·  " + str(N_PARTICLES) + " particles  ·  PLD " +
        str(PLD_DAYS) + " days  ·  reproductive season Jan–Jun",
        fontsize=11, fontweight="bold", color=site["color"], y=1.02)
    plt.tight_layout(pad=1.2)
    out = os.path.join(OUTDIR, "site_" + sc + "_6rep.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("        Saved: " + out)


# --- B: Per site — SUMMARY panel (all 6 replicates overlaid) ---
print("\n    [B] Per-site summary maps (all replicates overlaid)...")
for sc, site in SITES.items():
    print("        " + sc + "...")
    if HAS_CARTOPY:
        fig, ax = plt.subplots(figsize=(10, 8),
                               subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,      facecolor="#c8b89a", zorder=2)
        ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#3a2a10", zorder=4)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#999999", zorder=4)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.5, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([6, 9, 12, 15, 18])
        gl.ylocator = mticker.FixedLocator([37, 39, 41, 43, 45])
        gl.xlabel_style = {"size": 8, "color": "gray"}
        gl.ylabel_style = {"size": 8, "color": "gray"}
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#d0e8f5")
        ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_xticks([6, 9, 12, 15, 18]); ax.set_yticks([37, 39, 41, 43, 45])
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3, linestyle="--")

    for rep in REPLICATES:
        fpath = traj_files.get((sc, rep["name"]))
        if not fpath:
            continue
        ds2   = xr.open_zarr(fpath)
        lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
        ds2.close()
        draw_traj(ax, lons2, lats2, rep["color"], alpha=0.55, lw=0.8)

    for other_sc, other_site in SITES.items():
        if other_sc == sc:
            continue
        kw = dict(color=other_site["color"], ms=5, markeredgecolor="white",
                  markeredgewidth=0.8, alpha=0.5, zorder=7)
        if HAS_CARTOPY:
            ax.plot(other_site["lon"], other_site["lat"], "o",
                    transform=ccrs.PlateCarree(), **kw)
        else:
            ax.plot(other_site["lon"], other_site["lat"], "o", **kw)

    draw_marker(ax, site["lon"], site["lat"], site["color"],
                sc + " — " + site["name"], ms=11)

    rep_handles = [mpatches.Patch(color=r["color"],
                                  label=r["name"] + " (" + str(r["release"][1]).zfill(2) +
                                  "/" + str(r["release"][0]).zfill(2) + ")")
                   for r in REPLICATES]
    ax.legend(handles=rep_handles, loc="upper right", fontsize=9,
              title="Release date", framealpha=0.95, title_fontsize=9)

    ax.set_title(
        sc + " — " + site["name"] + "  |  All replicates overlaid\n"
        "Electra posidoniae Cyphonautes  ·  CMEMS NEMO-OPA 4.2 km  ·  " +
        str(N_PARTICLES) + " particles  ·  PLD " + str(PLD_DAYS) + " days",
        fontsize=11, fontweight="bold", color=site["color"])

    plt.tight_layout()
    out = os.path.join(OUTDIR, "site_" + sc + "_summary.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("        Saved: " + out)


# --- C: Overview all sites colored by site ---
print("\n    [C] Overview map...")
if HAS_CARTOPY:
    fig, ax = plt.subplots(figsize=(16, 10),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#c8b89a", zorder=2)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#3a2a10", zorder=4)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#999999", zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([6, 8, 10, 12, 14, 16, 18])
    gl.ylocator = mticker.FixedLocator([37, 38, 39, 40, 41, 42, 43, 44, 45])
    gl.xlabel_style = {"size": 8, "color": "gray"}
    gl.ylabel_style = {"size": 8, "color": "gray"}
else:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#d0e8f5")
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_xticks([6, 8, 10, 12, 14, 16, 18]); ax.set_yticks([37, 39, 41, 43, 45])
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3, linestyle="--")

for sc, site in SITES.items():
    for rep in REPLICATES:
        fpath = traj_files.get((sc, rep["name"]))
        if not fpath:
            continue
        ds2   = xr.open_zarr(fpath)
        lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
        ds2.close()
        draw_traj(ax, lons2, lats2, site["color"], alpha=0.25, lw=0.6)

for sc, site in SITES.items():
    draw_marker(ax, site["lon"], site["lat"], site["color"],
                sc + " " + site["name"], ms=10)

site_handles = [mpatches.Patch(color=s["color"], label=k + " — " + s["name"])
                for k, s in SITES.items()]
ax.legend(handles=site_handles, loc="upper right", fontsize=9,
          title="Site", framealpha=0.95, title_fontsize=9)
ax.set_title(
    "Electra posidoniae — Overview larval dispersal (all sites · all replicates)\n"
    "CMEMS NEMO-OPA 4.2 km  ·  " + str(N_PARTICLES) + " particles/site  ·  PLD " +
    str(PLD_DAYS) + " days  ·  colored by site",
    fontsize=11, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUTDIR, "overview_all_sites.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print("        Saved: " + out)


# --- D: Connectivity matrix ---
print("\n    [D] Connectivity matrix...")
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(conn, cmap="YlOrRd", vmin=0, vmax=100)
cbar = plt.colorbar(im, ax=ax, shrink=0.85,
                    label="% particles settling within " + str(int(SETTLE_KM)) + " km of destination")
cbar.ax.tick_params(labelsize=9)
site_labels = [sc + "\n" + SITES[sc]["name"] for sc in site_codes]
ax.set_xticks(range(n)); ax.set_xticklabels(site_labels, fontsize=10)
ax.set_yticks(range(n)); ax.set_yticklabels(site_labels, fontsize=10)
ax.set_xlabel("Destination site", fontsize=11, labelpad=8)
ax.set_ylabel("Source site", fontsize=11, labelpad=8)
ax.set_title(
    "Connectivity matrix — Electra posidoniae Cyphonautes\n"
    "% particles within " + str(int(SETTLE_KM)) + " km of destination  |  " +
    str(N_PARTICLES) + " particles/site  |  PLD " + str(PLD_DAYS) + " days  |  6 replicates Jan–Jun",
    fontsize=11, pad=10)
for i in range(n):
    for j in range(n):
        val = conn[i, j]
        ax.text(j, i, str(round(val, 1)) + "%", ha="center", va="center",
                fontsize=10, color="white" if val > 55 else "black",
                fontweight="bold" if i == j else "normal")
for tick, sc in zip(ax.get_xticklabels(), site_codes):
    tick.set_color(SITES[sc]["color"])
for tick, sc in zip(ax.get_yticklabels(), site_codes):
    tick.set_color(SITES[sc]["color"])
plt.tight_layout()
out = os.path.join(OUTDIR, "connectivity_matrix.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("        Saved: " + out)


# --- E: Dispersal boxplots ---
print("\n    [E] Dispersal boxplots...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
axes = axes.flatten()
rep_colors_list = [r["color"] for r in REPLICATES]

for s_idx, (sc, site) in enumerate(SITES.items()):
    ax = axes[s_idx]
    all_dists = []
    labels    = []
    for rep in REPLICATES:
        fpath = traj_files.get((sc, rep["name"]))
        if not fpath:
            all_dists.append(np.array([0])); labels.append(rep["name"])
            continue
        ds2   = xr.open_zarr(fpath)
        lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
        ds2.close()
        flons, flats = [], []
        for i in range(lons2.shape[0]):
            v = ~np.isnan(lons2[i])
            flons.append(lons2[i][v][-1] if v.any() else site["lon"])
            flats.append(lats2[i][v][-1] if v.any() else site["lat"])
        dists = haversine_km(site["lon"], site["lat"],
                             np.array(flons), np.array(flats))
        all_dists.append(dists)
        labels.append(rep["name"])

    bp = ax.boxplot(all_dists, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, col in zip(bp["boxes"], rep_colors_list):
        patch.set_facecolor(col); patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Dispersal distance (km)", fontsize=8)
    ax.set_title(sc + " — " + site["name"], fontsize=10,
                 fontweight="bold", color=site["color"])
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(SETTLE_KM, color="gray", linestyle=":", linewidth=1.2)
    ax.text(0.02, SETTLE_KM + 3, str(int(SETTLE_KM)) + " km",
            transform=ax.get_yaxis_transform(), fontsize=7, color="gray")

plt.suptitle(
    "Dispersal distance distribution — Electra posidoniae Cyphonautes\n"
    "PLD " + str(PLD_DAYS) + " days  ·  " + str(N_PARTICLES) +
    " particles/site  ·  boxes=IQR  ·  line=median  ·  dotted=" + str(int(SETTLE_KM)) + " km",
    fontsize=10, y=1.01)
plt.tight_layout()
out = os.path.join(OUTDIR, "dispersal_boxplot.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("        Saved: " + out)

print("\n" + "="*65)
print("Done! All results in: " + OUTDIR)
print("="*65)
print("\nFiles generated:")
print("  Simulations (36 zarr files):")
for sc in SITES:
    for rep in REPLICATES:
        print("    traj_" + sc + "_" + rep["name"] + ".zarr")
print("\n  Plots:")
for sc in SITES:
    print("    site_" + sc + "_6rep.png      — 6 panels (one per replicate)")
    print("    site_" + sc + "_summary.png   — all replicates overlaid")
print("    overview_all_sites.png         — all sites + all replicates")
print("    connectivity_matrix.png        — connectivity heatmap")
print("    dispersal_boxplot.png          — distance distribution")
print("    dispersal_statistics.csv       — full statistics table")
print("    connectivity_matrix.csv        — connectivity table")
