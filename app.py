import streamlit as st
import cdsapi
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import tempfile
from datetime import datetime
import io

# -------------------- Streamlit page --------------------
st.set_page_config(page_title="ERA5 General Archive Data Plotter", layout="wide")

# Regions (lon_w, lon_e, lat_s, lat_n). Longitudes may be outside [-180,180]
REGIONS = {
    "General": {
        "Continental United States": [-125, -66.5, 24.396308, 49.384358],
        "Europe": [-12.2, 49.4, 26.6, 74.3],
        "Middle East and South Asia": [27.9, 102.3, 1.8, 67.5],
        "East and Southeast Asia": [86.4, 160.8, -14.7, 50.9],
        "Australia and Oceania": [108.8, 191, -52.6, -5.8],
        "Northern Africa": [-20.5, 55.6, -4.2, 39.4],
        "Southern Africa": [2.8, 59.5, -39.5, -4.7],
        "Northern South America": [-83.5, -31.3, -24.2, 13.7],
        "Southern South America": [-86.4, -34.3, -58.4, -15.7],
        "Mainland Canada": [-128, -52, 40.6, 62.7],
        "Mexico and Central America": [-119, -56.1, 3.3, 35.7]
    },
    "Tropics": {
        "North Atlantic Basin": [-102.8, -7.9, 6, 57.6],
        "West Pacific Basin": [94.9, 183.5, -14.6, 56.1],       # crosses IDL
        "East Pacific Basin": [-161.4, -86.3, 3, 39],
        "Central Pacific Basin": [-188.8, -141.6, 2.4, 41.1],   # crosses IDL
        "Northern Indian Ocean Basin": [-317, -256.3, -5, 34],
        "South Indian Ocean Basin": [32.7, 125.4, -44.8, 3.5],
        "Australian Basin": [100, 192.7, -50.2, -1.9]
    }
}

# -------------------- Colormaps (match your scripts) --------------------
def create_custom_dewpoint_cmap():
    colors = [
        (152,109, 77),(150,108, 76),(148,107, 76),(146,106, 75),(144,105, 75),(142,104, 74),
        (140,102, 74),(138,101, 73),(136,100, 72),(134, 99, 72),(132, 98, 71),(130, 97, 71),
        (128, 96, 70),(126, 95, 70),(124, 94, 69),(122, 93, 68),(120, 91, 68),(118, 90, 67),
        (116, 89, 67),(114, 88, 66),(113, 87, 66),(109, 85, 64),(107, 84, 64),(105, 83, 63),
        (103, 82, 63),(101, 80, 62),( 99, 79, 61),( 97, 78, 61),( 95, 77, 60),( 93, 76, 60),
        ( 91, 75, 59),( 89, 74, 59),( 87, 73, 58),( 85, 72, 57),( 83, 71, 57),( 81, 69, 56),
        ( 79, 68, 56),( 77, 67, 55),( 75, 66, 55),( 73, 65, 54),( 71, 64, 54),( 69, 63, 53),
        ( 77, 67, 52),( 81, 71, 56),( 86, 76, 60),( 90, 80, 65),( 94, 85, 69),( 99, 89, 73),
        (103, 94, 77),(107, 98, 81),(112,103, 86),(116,107, 90),(120,112, 94),(125,116, 98),
        (129,121,103),(133,125,107),(138,130,111),(142,134,115),(146,139,119),(151,143,124),
        (155,148,128),(159,152,132),(164,157,137),(168,161,141),(173,166,145),(189,179,156),
        (189,179,156),(188,184,161),(193,188,165),(201,197,173),(201,197,173),(210,206,182),
        (223,220,194),(227,224,198),(231,229,202),(235,233,207),(240,238,211),(244,242,215),
        (230,245,230),(215,240,215),(200,234,200),(185,229,185),(170,223,170),(155,218,155),
        (140,213,140),(125,207,125),(110,202,110),( 95,196, 95),( 80,191, 80),( 65,186, 65),
        ( 48,174, 48),( 44,163, 44),( 39,153, 39),( 35,142, 35),( 30,131, 30),( 26,121, 26),
        ( 21,110, 21),( 17, 99, 17),( 12, 89, 12),(  8, 78,  8),( 97,163,175),( 88,150,160),
        ( 80,137,146),( 71,123,131),( 62,110,116),( 54, 97,102),( 45, 84, 87),( 36, 70, 72),
        ( 28, 57, 58),( 19, 44, 43),(102,102,154),( 96, 94,148),( 89, 86,142),( 83, 78,136),
        ( 77, 70,130),( 70, 62,124),( 64, 54,118),( 58, 46,112),( 51, 38,106),( 45, 30,100),
        (114, 64,113),(120, 69,115),(125, 75,117),(131, 80,118),(136, 86,120),(142, 91,122),
        (147, 97,124),(153,102,125),(158, 108,127),(164, 113,129)
    ]
    norm = [(r/255.0, g/255.0, b/255.0) for r,g,b in colors]
    return mcolors.LinearSegmentedColormap.from_list('custom_dewpoint', norm, N=256)

def cmap_500_wind():
    pw500speed_colors = [
        (230,244,255),(219,240,254),(209,235,254),(198,231,253),(188,227,253),(177,223,252),
        (167,219,252),(156,214,251),(146,210,251),(135,206,250),(132,194,246),(129,183,241),
        (126,171,237),(123,160,232),(121,148,228),(118,136,223),(115,125,219),(112,113,214),
        (109,102,210),(106, 90,205),(118, 96,207),(131,102,208),(143,108,210),(156,114,211),
        (168,120,213),(180,126,214),(193,132,216),(205,138,217),(218,144,219),(230,150,220),
        (227,144,217),(224,138,214),(221,132,211),(218,126,208),(215,120,205),(212,114,202),
        (209,108,199),(206,102,196),(203, 96,193),(200, 90,190),(196, 83,186),(192, 76,182),
        (188, 69,178),(184, 62,174),(180, 55,170),(176, 48,166),(172, 41,162),(168, 34,158),
        (164, 27,154),(160, 20,150),(164, 16,128),(168, 14,117),(172, 14,117),(176, 12,106),
        (180,  8, 95),(184,  6, 73),(188,  4, 62),(192,  2, 51),(200,  0, 40),(200,  0, 40),
        (202,  4, 42),(204,  8, 44),(208, 12, 44),(210, 20, 50),(212, 24, 52),(212, 24, 52),
        (214, 28, 54),(218, 36, 58),(220, 40, 60),(222, 44, 62),(224, 48, 64),(226, 52, 66),
        (228, 56, 68),(230, 60, 70),(232, 64, 72),(234, 68, 74),(236, 72, 76),(238, 76, 78),
        (240, 80, 80),(241, 96, 82),(242,112, 84),(243,128, 86),(244,144, 88),(245,160, 90),
        (246,176, 92),(247,192, 94),(248,208, 96),(249,224, 98),(250,240,100),(247,235, 97),
        (244,230, 94),(241,225, 91),(238,220, 88),(235,215, 85),(232,210, 82),(229,205, 79),
        (226,200, 76),(223,195, 73),(220,190, 70),(217,185, 67),(214,180, 64),(211,175, 61),
        (208,170, 58),(205,165, 55),(202,160, 52),(199,155, 49),(196,150, 46),(193,145, 43),
        (190,140, 40),(187,135, 37),(184,130, 34),(181,125, 31),(178,120, 28),(175,115, 25),
        (172,110, 22),(169,105, 19),(166,100, 16),(163, 95, 13),(160, 90, 10),(160, 90, 10)
    ]
    norm = [(r/255, g/255, b/255) for r,g,b in pw500speed_colors]
    return mcolors.LinearSegmentedColormap.from_list("wind500_cmap", norm)

def cmap_850_wind():
    pw850speed_colors = [
        (240,248,255),(219,240,254),(198,231,253),(177,223,252),(156,214,251),(135,206,250),
        (129,183,241),(123,160,232),(118,136,223),(112,113,214),(106, 90,205),(131,102,208),
        (156,114,211),(180,126,214),(205,138,217),(230,150,220),(224,138,214),(218,126,208),
        (212,114,202),(206,102,196),(200, 90,190),(192, 76,182),(184, 62,174),(176, 48,166),
        (168, 34,158),(160, 20,150),(168, 16,128),(176, 12,106),(184,  4, 62),(200,  0, 40),
        (200,  0, 40),(208, 16, 48),(212, 24, 52),(216, 32, 56),(220, 40, 60),(224, 48, 64),
        (228, 56, 68),(232, 64, 72),(236, 72, 76),(240, 80, 80),(242,112, 84),(244,144, 88),
        (246,176, 92),(248,208, 96),(250,240,100),(244,230, 94),(238,220, 88),(232,210, 82),
        (226,200, 76),(220,190, 70),(214,180, 64),(208,170, 58),(202,160, 52),(196,150, 46),
        (190,140, 40),(184,130, 34),(178,120, 28),(172,110, 22),(166,100, 16),(160, 90, 10)
    ]
    norm = [(r/255, g/255, b/255) for r,g,b in pw850speed_colors]
    return mcolors.LinearSegmentedColormap.from_list("wind850_cmap", norm)

def cmap_cp_white_to_jet():
    jet = plt.get_cmap('jet')
    colors = [(1,1,1)] * 10  # 0..0.1 mm white
    n_colors = 99
    jet_colors = [jet(i / n_colors) for i in range(n_colors)]
    colors.extend(jet_colors)
    return mcolors.LinearSegmentedColormap.from_list("custom_jet", colors, N=100)

def rbtop3():
    return mcolors.LinearSegmentedColormap.from_list("", [
        (0/140, "#000000"),
        (60/140, "#fffdfd"),
        (60/140, "#05fcfe"),
        (70/140, "#010071"),
        (80/140, "#00fe24"),
        (90/140, "#fbff2d"),
        (100/140,"#fd1917"),
        (110/140,"#000300"),
        (120/140,"#e1e4e5"),
        (120/140,"#eb6fc0"),
        (130/140,"#9b1f94"),
        (140/140,"#330f2f")
    ]).reversed()

# -------------------- Dateline-safe recentering --------------------
def mod360(x):
    return (np.asarray(x) % 360.0 + 360.0) % 360.0

def shortest_arc_mid(lw, le):
    w = mod360(lw); e = mod360(le)
    d = (e - w) % 360.0
    if d <= 180.0:
        center = (w + d / 2.0) % 360.0
        w_u, e_u = w, w + d
    else:
        d2 = 360.0 - d
        center = (e + d2 / 2.0) % 360.0
        w_u, e_u = w, w + 360.0 - d2
    return center, w_u, e_u

def build_projection_and_extent(lon_w, lon_e, lat_s, lat_n):
    s, n = (lat_s, lat_n) if lat_s <= lat_n else (lat_n, lat_s)
    center, w_u, e_u = shortest_arc_mid(lon_w, lon_e)

    def to_center_frame(lon):
        return ((lon - center + 180.0) % 360.0) - 180.0

    w_c = to_center_frame(w_u)
    e_c = to_center_frame(e_u)
    proj = ccrs.PlateCarree(central_longitude=float(center))
    extent_crs = ccrs.PlateCarree(central_longitude=float(center))
    extent = [w_c, e_c, s, n]
    return proj, extent_crs, extent, float(center)

def subset_lon_lat(lon_360, lat, data3d, lon_w, lon_e, lat_s, lat_n):
    lon_pm180 = ((lon_360 + 180.0) % 360.0) - 180.0
    s, n = (lat_s, lat_n) if lat_s <= lat_n else (lat_n, lat_s)
    center, w_u, e_u = shortest_arc_mid(lon_w, lon_e)
    cmp_axis = mod360(lon_pm180)  # 0..360 for comparison
    cmp_unwrapped = np.where(cmp_axis < w_u, cmp_axis + 360.0, cmp_axis)
    idx = np.where((cmp_unwrapped >= w_u) & (cmp_unwrapped <= e_u))[0]
    if idx.size == 0:
        idx = np.arange(lon_pm180.size)
    idx = idx[np.argsort(cmp_unwrapped[idx])]
    lon_sel = lon_pm180[idx]

    # latitude selection robust to descending arrays
    if lat[0] < lat[-1]:
        lat_mask = (lat >= s) & (lat <= n)
    else:
        lat_mask = (lat <= n) & (lat >= s)
    lat_sel = lat[lat_mask]

    sub = data3d[:, lat_mask, :][:, :, idx]
    return lon_sel, lat_sel, sub

def ensure_increasing_axes(lon1d, lat1d, *fields):
    lon_up, lat_up = lon1d, lat1d
    out = list(fields)
    if lon_up[0] > lon_up[-1]:
        lon_up = lon_up[::-1]
        out = [np.ascontiguousarray(f[..., ::-1]) for f in out]
    if lat_up[0] > lat_up[-1]:
        lat_up = lat_up[::-1]
        out = [np.ascontiguousarray(f[:, ::-1, :]) for f in out]
    return (lon_up, lat_up, *out)

def to_center_frame_vec(lon_pm180, center_deg):
    return ((lon_pm180 - center_deg + 180.0) % 360.0) - 180.0

# -------------------- Zoom-aware density & intervals --------------------
def auto_plot_params(extent, nx, ny):
    w, e, s, n = extent
    lon_span = e - w
    lat_span = abs(n - s)
    span = max(lon_span, lat_span)

    if span >= 120:
        desired_x = 28; barb_len = 5; barb_min_stride = 9
        mslp_lw = 0.95; coast_lw = 0.9; border_lw = 0.75; state_lw = 0.5; cint = 3
    elif span >= 60:
        desired_x = 45; barb_len = 6; barb_min_stride = 8
        mslp_lw = 1.0; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 2
    elif span >= 30:
        desired_x = 65; barb_len = 6; barb_min_stride = 7
        mslp_lw = 1.05; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 3
    else:
        desired_x = 85; barb_len = 6; barb_min_stride = 6
        mslp_lw = 1.1; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 4

    stride_x = max(1, min(14, max(nx // desired_x, barb_min_stride)))
    stride_y = max(1, min(14, max(ny // int(desired_x / 1.6), barb_min_stride)))

    return {
        'stride_y': stride_y, 'stride_x': stride_x, 'barb_len': barb_len,
        'mslp_lw': mslp_lw, 'coast_lw': coast_lw, 'border_lw': border_lw,
        'state_lw': state_lw, 'cint': cint
    }

# -------------------- Time helper --------------------
def read_valid_time(ds):
    var = ds.variables.get('valid_time') or ds.variables.get('time')
    if var is None:
        return "Unknown valid time"
    time_unit = getattr(var, 'units', None)
    time_calendar = getattr(var, 'calendar', 'standard')
    try:
        date = nc.num2date(var[:][0], units=time_unit, calendar=time_calendar)
        return date.strftime("%B %d, %Y - %H:%M UTC")
    except Exception:
        return "Unknown valid time"

# -------------------- Data fetch helpers --------------------
def cds_retrieve_single(date_input, api_key, variables):
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)
    req = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": date_input[:4],
        "month": date_input[4:6],
        "day": date_input[6:8],
        "time": date_input[8:] + ":00",
        "format": "netcdf"
    }
    tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
    target = tmp.name; tmp.close()
    c.retrieve("reanalysis-era5-single-levels", req, target)
    return target

def cds_retrieve_pl(date_input, api_key, pressure_level, variables):
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)
    req = {
        "product_type": "reanalysis",
        "pressure_level": [str(pressure_level)],
        "variable": variables,
        "year": date_input[:4],
        "month": date_input[4:6],
        "day": date_input[6:8],
        "time": date_input[8:] + ":00",
        "format": "netcdf"
    }
    tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
    target = tmp.name; tmp.close()
    c.retrieve("reanalysis-era5-pressure-levels", req, target)
    return target

# -------------------- Plotters --------------------
def plot_base_map(ax, params):
    ax.set_facecolor('#C0C0C0')
    ax.add_feature(cfeature.COASTLINE, edgecolor='#7f7f7f', linewidth=params['coast_lw'])
    ax.add_feature(cfeature.BORDERS, edgecolor='#7f7f7f', linestyle=':', linewidth=params['border_lw'])
    try:
        ax.add_feature(cfeature.STATES, edgecolor='#9e9e9e', linestyle=':', linewidth=params['state_lw'])
    except Exception:
        pass

def add_barbs(ax, lon_plot, lat_sel, u, v, params, transform):
    LON2, LAT2 = np.meshgrid(lon_plot, lat_sel)
    si = params['stride_y']; sj = params['stride_x']
    ax.barbs(
        LON2[::si, ::sj], LAT2[::si, ::sj],
        u[0, ::si, ::sj], v[0, ::si, ::sj],
        length=params['barb_len'], transform=transform
    )

# -------------------- Main renderer --------------------
def generate_visualization(year, month, day, hour, region_coords, api_key, product):
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"
    lon_w, lon_e, lat_s, lat_n = region_coords

    # Always get single levels core for IDL-safe lon/lat grid + winds/title when needed
    single_vars = [
        'mean_sea_level_pressure','2m_dewpoint_temperature','2m_temperature',
        '10m_u_component_of_wind','10m_v_component_of_wind'
    ]
    single_file = cds_retrieve_single(date_input, api_key, single_vars)
    ds = nc.Dataset(single_file)

    # Read common single-level fields
    mslp = ds.variables['msl'][:] / 100.0
    d2m  = ds.variables['d2m'][:]
    u10  = ds.variables['u10'][:]
    v10  = ds.variables['v10'][:]
    lon0 = ds.variables['longitude'][:]
    lat  = ds.variables['latitude'][:]

    # Enforce strictly increasing longitudes (0..360)
    order = np.argsort(lon0)
    lon_360 = lon0[order]
    mslp = mslp[:, :, order]
    d2m  = d2m[:,  :, order]
    u10  = u10[:,  :, order]
    v10  = v10[:,  :, order]

    # Build map/projection for region (IDL-proof)
    proj, extent_crs, extent_centered, center_deg = build_projection_and_extent(lon_w, lon_e, lat_s, lat_n)

    # Subset common grids to region
    lon_sel, lat_sel, mslp_sub = subset_lon_lat(lon_360, lat, mslp, lon_w, lon_e, lat_s, lat_n)
    _,       _,       d2m_sub  = subset_lon_lat(lon_360, lat, d2m,  lon_w, lon_e, lat_s, lat_n)
    _,       _,       u10_sub  = subset_lon_lat(lon_360, lat, u10,  lon_w, lon_e, lat_s, lat_n)
    _,       _,       v10_sub  = subset_lon_lat(lon_360, lat, v10,  lon_w, lon_e, lat_s, lat_n)

    # Center longitudes to the map frame and sort axes
    lon_plot = to_center_frame_vec(((lon_sel + 180.0) % 360.0) - 180.0, center_deg)
    lon_plot = np.where(np.isclose(np.diff(np.r_[lon_plot[0]-1e-8, lon_plot]), 0.0), lon_plot + 1e-8, lon_plot)

    lon_plot, lat_sel, mslp_sub, d2m_sub, u10_sub, v10_sub = ensure_increasing_axes(
        lon_plot, lat_sel, mslp_sub, d2m_sub, u10_sub, v10_sub
    )

    data_crs_centered = ccrs.PlateCarree(central_longitude=center_deg)
    date_str = read_valid_time(ds)

    # Figure / axes
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': proj})
    ax.set_extent(extent_centered, crs=extent_crs)
    params = auto_plot_params(extent_centered, nx=lon_plot.size, ny=lat_sel.size)
    plot_base_map(ax, params)

    # ---------- Product branches ----------
    if product == "Dewpoint, MSLP, and wind barbs":
        dewpoint_f = (d2m_sub - 273.15) * 9/5 + 32
        cmap = create_custom_dewpoint_cmap()

        # Isobars
        mslp0 = np.ascontiguousarray(mslp_sub[0, :, :])
        cint = params['cint']
        mmin = np.floor(np.nanmin(mslp0) / cint) * cint
        mmax = np.ceil(np.nanmax(mslp0) / cint) * cint
        levels = np.arange(mmin, mmax + cint, cint)
        if levels.size > 60:
            levels = levels[::int(np.ceil(levels.size / 60))]

        ax.contour(lon_plot, lat_sel, mslp0, levels=levels, colors='black',
                   linewidths=params['mslp_lw'], transform=data_crs_centered)

        # Filled dewpoint
        cf = ax.contourf(lon_plot, lat_sel, dewpoint_f[0, :, :],
                         levels=np.linspace(-40, 90, 256), cmap=cmap, extend='both',
                         transform=data_crs_centered)

        add_barbs(ax, lon_plot, lat_sel, u10_sub, v10_sub, params, data_crs_centered)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('2m Dewpoint Temperature (°F)')
        title = f'ERA5 Pressure, Dewpoint, and 10 m Wind — Valid: {date_str}'

    elif product == "Surface-based CAPE and wind barbs":
        # Retrieve SBCAPE (single-levels 'convective_available_potential_energy' -> short name 'cape')
        cape_file = cds_retrieve_single(date_input, api_key, ['convective_available_potential_energy'])
        dsc = nc.Dataset(cape_file)
        # Try common short names in case CDS returns variant
        cape_var = dsc.variables.get('cape') or dsc.variables.get('convective_available_potential_energy')
        if cape_var is None:
            # Fallback to first 2D var
            for k,v in dsc.variables.items():
                if getattr(v, 'dimensions', None) and len(v.dimensions) >= 2:
                    cape_var = v; break
        cape = cape_var[:]  # J/kg
        lonp = dsc.variables['longitude'][:]; latp = dsc.variables['latitude'][:]
        order_c = np.argsort(lonp); lon_c = lonp[order_c]
        cape = cape[:, :, order_c]

        lon_sel_c, lat_sel_c, cape_sub = subset_lon_lat(lon_c, latp, cape, lon_w, lon_e, lat_s, lat_n)
        lon_plot_c = to_center_frame_vec(((lon_sel_c + 180.0) % 360.0) - 180.0, center_deg)
        lon_plot_c, lat_sel_c, cape_sub = ensure_increasing_axes(lon_plot_c, lat_sel_c, cape_sub)

        # Clip 0..7000 and plot with turbo
        cape_clip = np.clip(cape_sub[0, :, :], 0, 7000)
        turbo = plt.get_cmap('turbo')
        cf = ax.contourf(lon_plot_c, lat_sel_c, cape_clip,
                         levels=np.linspace(0, 7000, 71), cmap=turbo, extend='max',
                         transform=data_crs_centered)

        # 10 m barbs from earlier
        add_barbs(ax, lon_plot, lat_sel, u10_sub, v10_sub, params, data_crs_centered)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('Surface-based CAPE (J/kg) — clipped at 7000')
        title = f'ERA5 SBCAPE (turbo, 0–7000) + 10 m Wind — Valid: {date_str}'
        dsc.close()
        os.remove(cape_file)

    elif product in ("500 mb wind and height", "850 mb wind and height"):
        level = 500 if product.startswith("500") else 850
        pl_file = cds_retrieve_pl(date_input, api_key, level,
                                  ['geopotential','u_component_of_wind','v_component_of_wind'])
        dsp = nc.Dataset(pl_file)
        z  = dsp.variables['z'][0, 0, ...] / 9.80665  # m
        z_dm = z / 10.0                               # dam
        u   = dsp.variables['u'][0, 0, ...]
        v   = dsp.variables['v'][0, 0, ...]
        lonp = dsp.variables['longitude'][:]; latp = dsp.variables['latitude'][:]

        order_p = np.argsort(lonp); lon_p = lonp[order_p]
        z_dm = z_dm[:, order_p]
        u    = u[:,  order_p]
        v    = v[:,  order_p]

        # Make as 3D [time, lat, lon] for subsetting convenience
        z3 = z_dm[np.newaxis, ...]; u3 = u[np.newaxis, ...]; v3 = v[np.newaxis, ...]
        lon_sel_p, lat_sel_p, z_sub = subset_lon_lat(lon_p, latp, z3, lon_w, lon_e, lat_s, lat_n)
        _,          _,          u_sub = subset_lon_lat(lon_p, latp, u3, lon_w, lon_e, lat_s, lat_n)
        _,          _,          v_sub = subset_lon_lat(lon_p, latp, v3, lon_w, lon_e, lat_s, lat_n)

        lon_plot_p = to_center_frame_vec(((lon_sel_p + 180.0) % 360.0) - 180.0, center_deg)
        lon_plot_p, lat_sel_p, z_sub, u_sub, v_sub = ensure_increasing_axes(
            lon_plot_p, lat_sel_p, z_sub, u_sub, v_sub
        )

        # Wind speed (kts)
        wspd_ms = np.sqrt(u_sub[0]**2 + v_sub[0]**2)
        wspd_kts = wspd_ms * 1.94384

        if level == 500:
            cmap_ws = cmap_500_wind()
            ws_levels = np.arange(20, 141, 1)
            h_levels  = np.arange(480, 600, 6)
        else:
            cmap_ws = cmap_850_wind()
            ws_levels = np.arange(20, 81, 1)
            h_levels  = np.arange(120, 180, 3)

        cf = ax.contourf(lon_plot_p, lat_sel_p, wspd_kts, levels=ws_levels, cmap=cmap_ws,
                         extend='both', transform=data_crs_centered)
        cs = ax.contour(lon_plot_p, lat_sel_p, z_sub[0], levels=h_levels, colors='black',
                        linewidths=1.0, transform=data_crs_centered)
        ax.clabel(cs, inline=True, fontsize=8, colors='black', fmt='%d')

        # Wind barbs (at level winds)
        LON2, LAT2 = np.meshgrid(lon_plot_p, lat_sel_p)
        si = params['stride_y']; sj = params['stride_x']
        ax.barbs(LON2[::si, ::sj], LAT2[::si, ::sj], u_sub[0, ::si, ::sj], v_sub[0, ::si, ::sj],
                 length=params['barb_len'], transform=data_crs_centered)

        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label(f'{level} mb Wind Speed (kt)')
        title = f'ERA5 {level} mb Wind Speed/Heights — Valid: {date_str}'

        dsp.close(); os.remove(pl_file)

    elif product == "Convective precipitation":
        cp_file = cds_retrieve_single(date_input, api_key, ['convective_precipitation'])
        dscp = nc.Dataset(cp_file)
        # variable short name usually 'cp'
        cpv = dscp.variables.get('cp') or dscp.variables.get('convective_precipitation')
        cp = cpv[:]  # meters
        lonp = dscp.variables['longitude'][:]; latp = dscp.variables['latitude'][:]
        order_cp = np.argsort(lonp); lon_cp = lonp[order_cp]
        cp = cp[:, :, order_cp]
        lon_sel_cp, lat_sel_cp, cp_sub = subset_lon_lat(lon_cp, latp, cp, lon_w, lon_e, lat_s, lat_n)
        lon_plot_cp = to_center_frame_vec(((lon_sel_cp + 180.0) % 360.0) - 180.0, center_deg)
        lon_plot_cp, lat_sel_cp, cp_sub = ensure_increasing_axes(lon_plot_cp, lat_sel_cp, cp_sub)

        cp_mm = cp_sub[0] * 1000.0
        cmap = cmap_cp_white_to_jet()
        cf = ax.contourf(lon_plot_cp, lat_sel_cp, cp_mm,
                         levels=np.linspace(0, 10, 51), cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=10),
                         extend='max', transform=data_crs_centered)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('Convective Precipitation (mm)')
        title = f'ERA5 Convective Precipitation — Valid: {date_str}'
        dscp.close(); os.remove(cp_file)

    elif product == "Cloud top (simulated satellite)":
        ct_file = cds_retrieve_single(date_input, api_key, ['mean_top_net_long_wave_radiation_flux'])
        dsct = nc.Dataset(ct_file)
        # try multiple keys commonly seen
        mt = (dsct.variables.get('avg_tnlwrf') or
              dsct.variables.get('mtnlwrf') or
              dsct.variables.get('tnlwrf') or
              next((v for k,v in dsct.variables.items() if k not in ('latitude','longitude','time')), None))
        mtnlwrf = mt[:]
        lonp = dsct.variables['longitude'][:]; latp = dsct.variables['latitude'][:]
        order_ct = np.argsort(lonp); lon_ct = lonp[order_ct]
        mtnlwrf = mtnlwrf[:, :, order_ct]
        lon_sel_ct, lat_sel_ct, mt_sub = subset_lon_lat(lon_ct, latp, mtnlwrf, lon_w, lon_e, lat_s, lat_n)
        lon_plot_ct = to_center_frame_vec(((lon_sel_ct + 180.0) % 360.0) - 180.0, center_deg)
        lon_plot_ct, lat_sel_ct, mt_sub = ensure_increasing_axes(lon_plot_ct, lat_sel_ct, mt_sub)

        # Stefan–Boltzmann to cloud-top temperature (°C)
        sigma = 5.670367e-8
        olr = np.abs(mt_sub[0])   # W/m^2
        t_k = (olr / sigma) ** 0.25
        t_c = t_k - 273.15

        cmap = rbtop3()
        cf = ax.contourf(lon_plot_ct, lat_sel_ct, t_c, levels=np.arange(-100, 41, 2),
                         cmap=cmap, extend='both', transform=data_crs_centered)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('Cloud Top Temperature (°C)')
        title = f'ERA5 Cloud Top (Simulated IR) — Valid: {date_str}'
        dsct.close(); os.remove(ct_file)

    else:
        # Fallback to default dewpoint/MSLP/wind
        dewpoint_f = (d2m_sub - 273.15) * 9/5 + 32
        cmap = create_custom_dewpoint_cmap()
        mslp0 = np.ascontiguousarray(mslp_sub[0, :, :])
        cint = params['cint']
        mmin = np.floor(np.nanmin(mslp0) / cint) * cint
        mmax = np.ceil(np.nanmax(mslp0) / cint) * cint
        levels = np.arange(mmin, mmax + cint, cint)
        ax.contour(lon_plot, lat_sel, mslp0, levels=levels, colors='black',
                   linewidths=params['mslp_lw'], transform=data_crs_centered)
        cf = ax.contourf(lon_plot, lat_sel, dewpoint_f[0, :, :],
                         levels=np.linspace(-40, 90, 256), cmap=cmap, extend='both',
                         transform=data_crs_centered)
        add_barbs(ax, lon_plot, lat_sel, u10_sub, v10_sub, params, data_crs_centered)
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('2m Dewpoint Temperature (°F)')
        title = f'ERA5 Pressure, Dewpoint, and 10 m Wind — Valid: {date_str}'

    ax.set_title(title + "\nPlotted by Sekai Chandra (@Sekai_WX)")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)

    ds.close()
    plt.close(fig)
    os.remove(single_file)

    return buffer

# -------------------- Streamlit UI --------------------
st.title("ERA5 General Archive Data Plotter")

# Get API key from secrets
try:
    api_key = st.secrets["CDS_API_KEY"]
except KeyError:
    st.error("CDS API key not found in secrets. Please configure your API key.")
    st.stop()

# Date inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.number_input("Year", min_value=1940, max_value=datetime.now().year, value=2023)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
with col4:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=12)

# Region + Product + Generate (second row)
col5, col6, col7 = st.columns([1,1,0.6])
with col5:
    region_options = []
    for category, regions in REGIONS.items():
        for region_name in regions.keys():
            region_options.append(f"{category}: {region_name}")
    selected_region = st.selectbox("Select Region", region_options)

with col6:
    product = st.selectbox(
        "Product",
        [
            "Dewpoint, MSLP, and wind barbs",
            "Surface-based CAPE and wind barbs",
            "500 mb wind and height",
            "850 mb wind and height",
            "Convective precipitation",
            "Cloud top (simulated satellite)"
        ],
        index=0
    )
with col7:
    generate_button = st.button("Generate", type="primary", help="Generate the ERA5 visualization")

# Run
if generate_button:
    category, region_name = selected_region.split(": ", 1)
    region_coords = REGIONS[category][region_name]
    try:
        with st.spinner("Downloading ERA5 data and generating visualization..."):
            image_buffer = generate_visualization(year, month, day, hour, region_coords, api_key, product)
        st.success("Visualization generated successfully!")
        st.image(image_buffer, caption=f"{product} • {year}-{month:02d}-{day:02d} {hour:02d} UTC • {region_name}")
        st.download_button(
            label="Download Image",
            data=image_buffer,
            file_name=f"ERA5_{product.replace(' ','_').replace('/','-')}_{year}{month:02d}{day:02d}{hour:02d}_{region_name.replace(' ', '_')}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.info("Make sure the date/time is valid and the API service is available.")
