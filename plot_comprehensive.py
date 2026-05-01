import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import xarray as xr

TRAJ_DIR = os.environ.get("TRAJ_DIR", os.path.join(os.getcwd(), "results"))
OUTDIR   = os.environ.get("OUTDIR",   os.path.join(os.getcwd(), "results"))
os.makedirs(OUTDIR, exist_ok=True)

SITES = {
    "GE": {"lon":  8.90, "lat": 44.40, "name": "Genova",        "color": "#e53935"},
    "EB": {"lon": 10.20, "lat": 43.50, "name": "Elba/Livorno",  "color": "#fb8c00"},
    "NA": {"lon": 13.96, "lat": 40.73, "name": "Ischia/Napoli", "color": "#8e24aa"},
    "TA": {"lon": 17.14, "lat": 40.44, "name": "Taranto",       "color": "#1e88e5"},
    "PA": {"lon": 13.40, "lat": 38.10, "name": "Palermo",       "color": "#dfeb07"},
    "MS": {"lon": 15.50, "lat": 38.20, "name": "Messina",       "color": "#43a047"},
}

REPLICATES = [
    {"name": "Jan", "release": (1, 1),  "color": "#1565c0", "label": "January\n01/01"},
    {"name": "Feb", "release": (2, 2),  "color": "#0288d1", "label": "February\n02/02"},
    {"name": "Mar", "release": (3, 2),  "color": "#00897b", "label": "March\n02/03"},
    {"name": "Apr", "release": (4, 2),  "color": "#558b2f", "label": "April\n02/04"},
    {"name": "May", "release": (5, 2),  "color": "#f9a825", "label": "May\n02/05"},
    {"name": "Jun", "release": (6, 2),  "color": "#e53935", "label": "June\n02/06"},
]

N_PARTICLES = 300
PLD_DAYS    = 60
EXTENT      = [5.5, 20.0, 36.5, 45.5]

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("Cartopy OK")
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not available")

# Load trajectory files
print("Loading trajectory files...")
traj_files = {}
for sc in SITES:
    for rep in REPLICATES:
        fname = os.path.join(TRAJ_DIR, "traj_" + sc + "_" + rep["name"] + ".zarr")
        if os.path.exists(fname):
            traj_files[(sc, rep["name"])] = fname
print("  Found " + str(len(traj_files)) + " files")


def setup_map(ax, extent=None):
    ext = extent if extent else EXTENT
    if HAS_CARTOPY:
        ax.set_extent(ext, crs=ccrs.PlateCarree())
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
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([6, 9, 12, 15, 18]); ax.set_yticks([37, 39, 41, 43, 45])
        ax.tick_params(labelsize=6); ax.grid(True, alpha=0.3, linestyle="--")


def draw_traj(ax, lons2, lats2, color, alpha=0.55, lw=0.75):
    for i in range(lons2.shape[0]):
        v = ~np.isnan(lons2[i])
        if v.sum() > 1:
            kw = dict(color=color, alpha=alpha, linewidth=lw, zorder=5)
            if HAS_CARTOPY:
                ax.plot(lons2[i][v], lats2[i][v], transform=ccrs.PlateCarree(), **kw)
            else:
                ax.plot(lons2[i][v], lats2[i][v], **kw)


def draw_marker(ax, lon, lat, color, label="", ms=9, fs=9, offset=(0.25, 0.18)):
    kw_p = dict(color=color, ms=ms, markeredgecolor="white", markeredgewidth=1.3, zorder=8)
    kw_t = dict(fontsize=fs, fontweight="bold", color=color, zorder=9)
    if HAS_CARTOPY:
        ax.plot(lon, lat, "o", transform=ccrs.PlateCarree(), **kw_p)
        if label:
            ax.text(lon + offset[0], lat + offset[1], label,
                    transform=ccrs.PlateCarree(), **kw_t)
    else:
        ax.plot(lon, lat, "o", **kw_p)
        if label:
            ax.text(lon + offset[0], lat + offset[1], label, **kw_t)


def draw_all_other_sites(ax, exclude_sc):
    for other_sc, other_site in SITES.items():
        if other_sc == exclude_sc:
            continue
        kw = dict(color=other_site["color"], ms=4, markeredgecolor="white",
                  markeredgewidth=0.7, alpha=0.5, zorder=7)
        if HAS_CARTOPY:
            ax.plot(other_site["lon"], other_site["lat"], "o",
                    transform=ccrs.PlateCarree(), **kw)
        else:
            ax.plot(other_site["lon"], other_site["lat"], "o", **kw)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FIGURE: for each site — 2 rows
#   Row 1: 6 monthly panels (Jan–Jun), trajectories in month color
#   Row 2: 1 summary panel (all months overlaid) + 1 dispersal barplot
#           + connectivity bar to other sites
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating comprehensive per-site figures...")

for sc, site in SITES.items():
    print("  Site: " + sc + " — " + site["name"])

    # Preload all trajectory data for this site
    site_trajs = {}
    for rep in REPLICATES:
        fpath = traj_files.get((sc, rep["name"]))
        if fpath:
            ds2 = xr.open_zarr(fpath)
            site_trajs[rep["name"]] = {
                "lons": ds2["lon"].values,
                "lats": ds2["lat"].values,
            }
            ds2.close()

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 14))
    gs  = gridspec.GridSpec(2, 7, figure=fig,
                            hspace=0.35, wspace=0.25,
                            left=0.03, right=0.97,
                            top=0.90, bottom=0.06)

    # Row 1: 6 monthly panels
    month_axes = []
    proj_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    for r_idx in range(6):
        ax = fig.add_subplot(gs[0, r_idx], **proj_kw)
        month_axes.append(ax)

    # Row 2, col 0-2: summary map (spans 3 columns)
    sum_ax = fig.add_subplot(gs[1, 0:3], **proj_kw)

    # Row 2, col 3-4: dispersal boxplot (spans 2 columns)
    box_ax = fig.add_subplot(gs[1, 3:5])

    # Row 2, col 5-6: retained % bar chart (spans 2 columns)
    ret_ax = fig.add_subplot(gs[1, 5:7])

    # ── Row 1: Monthly panels ─────────────────────────────────────────────
    for r_idx, rep in enumerate(REPLICATES):
        ax = month_axes[r_idx]
        setup_map(ax)
        ax.set_title(rep["label"], fontsize=9, fontweight="bold",
                     color=rep["color"], pad=3)

        if rep["name"] in site_trajs:
            draw_traj(ax, site_trajs[rep["name"]]["lons"],
                      site_trajs[rep["name"]]["lats"],
                      rep["color"], alpha=0.65, lw=0.85)

        draw_all_other_sites(ax, sc)
        draw_marker(ax, site["lon"], site["lat"], site["color"], sc, ms=9, fs=8)

    # ── Row 2 left: Summary map ────────────────────────────────────────────
    setup_map(sum_ax)
    sum_ax.set_title("All months overlaid — " + sc + " " + site["name"],
                     fontsize=10, fontweight="bold", color=site["color"], pad=4)

    for rep in REPLICATES:
        if rep["name"] in site_trajs:
            draw_traj(sum_ax, site_trajs[rep["name"]]["lons"],
                      site_trajs[rep["name"]]["lats"],
                      rep["color"], alpha=0.5, lw=0.75)

    draw_all_other_sites(sum_ax, sc)
    draw_marker(sum_ax, site["lon"], site["lat"], site["color"],
                sc + " " + site["name"], ms=11, fs=9)

    rep_handles = [mpatches.Patch(color=r["color"], label=r["label"].replace("\n", " "))
                   for r in REPLICATES]
    sum_ax.legend(handles=rep_handles, loc="lower left", fontsize=7,
                  title="Release month", framealpha=0.92, title_fontsize=7,
                  ncol=2)

    # ── Row 2 middle: Dispersal boxplot ────────────────────────────────────
    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        a = np.sin(np.radians(lat2-lat1)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(np.radians(lon2-lon1)/2)**2
        return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    all_dists = []
    labels    = []
    medians   = []
    for rep in REPLICATES:
        if rep["name"] not in site_trajs:
            all_dists.append(np.array([0])); labels.append(rep["name"]); medians.append(0)
            continue
        lons2 = site_trajs[rep["name"]]["lons"]
        lats2 = site_trajs[rep["name"]]["lats"]
        flons, flats = [], []
        for i in range(lons2.shape[0]):
            v = ~np.isnan(lons2[i])
            flons.append(lons2[i][v][-1] if v.any() else site["lon"])
            flats.append(lats2[i][v][-1] if v.any() else site["lat"])
        dists = haversine_km(site["lon"], site["lat"],
                             np.array(flons), np.array(flats))
        all_dists.append(dists)
        labels.append(rep["name"])
        medians.append(float(np.median(dists)))

    bp = box_ax.boxplot(all_dists, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, rep in zip(bp["boxes"], REPLICATES):
        patch.set_facecolor(rep["color"]); patch.set_alpha(0.8)

    box_ax.set_xticklabels(labels, fontsize=9)
    box_ax.set_ylabel("Dispersal distance (km)", fontsize=9)
    box_ax.set_title("Dispersal distance distribution", fontsize=10, fontweight="bold")
    box_ax.grid(axis="y", alpha=0.3, linestyle="--")
    box_ax.axhline(50, color="gray", linestyle=":", linewidth=1.2)
    box_ax.text(0.01, 52, "50 km", transform=box_ax.get_yaxis_transform(),
                fontsize=7, color="gray")

    # Annotate medians
    for i, (med, rep) in enumerate(zip(medians, REPLICATES)):
        box_ax.text(i + 1, med + 3, str(round(med, 0)) + " km",
                    ha="center", va="bottom", fontsize=7, color=rep["color"],
                    fontweight="bold")

    # ── Row 2 right: Retained % per month ─────────────────────────────────
    retained = []
    for rep in REPLICATES:
        if rep["name"] not in site_trajs:
            retained.append(0); continue
        lons2 = site_trajs[rep["name"]]["lons"]
        lats2 = site_trajs[rep["name"]]["lats"]
        flons, flats = [], []
        for i in range(lons2.shape[0]):
            v = ~np.isnan(lons2[i])
            flons.append(lons2[i][v][-1] if v.any() else site["lon"])
            flats.append(lats2[i][v][-1] if v.any() else site["lat"])
        dists = haversine_km(site["lon"], site["lat"],
                             np.array(flons), np.array(flats))
        retained.append(float(np.mean(dists < 50) * 100))

    x = np.arange(len(REPLICATES))
    bars = ret_ax.bar(x, retained, color=[r["color"] for r in REPLICATES],
                      alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)
    ret_ax.set_xticks(x)
    ret_ax.set_xticklabels(labels, fontsize=9)
    ret_ax.set_ylabel("% particles retained within 50 km", fontsize=9)
    ret_ax.set_ylim(0, 110)
    ret_ax.set_title("Local retention by month", fontsize=10, fontweight="bold")
    ret_ax.grid(axis="y", alpha=0.3, linestyle="--")
    ret_ax.axhline(50, color="gray", linestyle=":", linewidth=1)

    for bar, val in zip(bars, retained):
        ret_ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                    str(round(val, 1)) + "%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    # ── Supertitle ────────────────────────────────────────────────────────
    fig.suptitle(
        sc + " — " + site["name"] + "   |   Electra posidoniae Cyphonautes — Larval dispersal\n"
        "CMEMS NEMO-OPA 4.2 km  ·  " + str(N_PARTICLES) +
        " particles  ·  PLD " + str(PLD_DAYS) + " days  ·  reproductive season Jan–Jun",
        fontsize=13, fontweight="bold", color=site["color"], y=0.97)

    out = os.path.join(OUTDIR, "comprehensive_" + sc + ".png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("    Saved: " + out)


# ═══════════════════════════════════════════════════════════════════════════
# SEASONAL INTERPRETATION PANEL — overview map with month color coding
# Shows which month drives which pattern, all sites together
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating seasonal interpretation overview...")

if HAS_CARTOPY:
    fig, axes = plt.subplots(2, 3, figsize=(22, 14),
                             subplot_kw={"projection": ccrs.PlateCarree()})
else:
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.flatten()

for r_idx, rep in enumerate(REPLICATES):
    ax = axes[r_idx]
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
        gl.xlabel_style = {"size": 7, "color": "gray"}
        gl.ylabel_style = {"size": 7, "color": "gray"}
    else:
        ax.set_facecolor("#d0e8f5")
        ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_xticks([6, 9, 12, 15, 18]); ax.set_yticks([37, 39, 41, 43, 45])
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3, linestyle="--")

    ax.set_title(rep["label"].replace("\n", " — ") + "  (release + 60 days)",
                 fontsize=11, fontweight="bold", color=rep["color"], pad=5)

    for sc, site in SITES.items():
        fpath = traj_files.get((sc, rep["name"]))
        if not fpath:
            continue
        ds2   = xr.open_zarr(fpath)
        lons2 = ds2["lon"].values; lats2 = ds2["lat"].values
        ds2.close()
        draw_traj(ax, lons2, lats2, site["color"], alpha=0.55, lw=0.8)

    for sc, site in SITES.items():
        draw_marker(ax, site["lon"], site["lat"], site["color"], sc, ms=8, fs=8)

    site_handles = [mpatches.Patch(color=s["color"], label=k + " " + s["name"])
                    for k, s in SITES.items()]
    ax.legend(handles=site_handles, loc="upper right", fontsize=7,
              title="Site", framealpha=0.92, title_fontsize=7)

fig.suptitle(
    "Electra posidoniae Cyphonautes — Seasonal larval dispersal patterns\n"
    "CMEMS NEMO-OPA 4.2 km  ·  " + str(N_PARTICLES) +
    " particles/site  ·  PLD " + str(PLD_DAYS) + " days  ·  colored by site",
    fontsize=13, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUTDIR, "seasonal_patterns_overview.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print("    Saved: " + out)

print("\nDone! Files generated:")
for sc in SITES:
    print("  comprehensive_" + sc + ".png   — 6 monthly panels + summary + stats")
print("  seasonal_patterns_overview.png — 6 panels, one per month, all sites")
