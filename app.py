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
st.set_page_config(page_title="ERA5 Weather Visualization", layout="wide")

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

# -------------------- Colormap --------------------
def create_custom_colormap():
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
        (147, 97,124),(153,102,125),(158,108,127),(164,113,129)
    ]
    norm_colors = [(r/255.0, g/255.0, b/255.0) for (r, g, b) in colors]
    return mcolors.LinearSegmentedColormap.from_list('custom_dewpoint', norm_colors, N=256)

# -------------------- Dateline-safe math by recentering --------------------
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
    e_c = to_center_frame(e_u)  # guaranteed w_c < e_c
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

# ---- NEW: recenter longitudes for plotting to the map's central_longitude ----
def to_center_frame_vec(lon_pm180, center_deg):
    """Convert standard lon in [-180,180) to the centered frame (-180..180] for plotting."""
    return ((lon_pm180 - center_deg + 180.0) % 360.0) - 180.0

# -------------------- Zoom-aware density & intervals --------------------
def auto_plot_params(extent, nx, ny):
    """
    Choose barb thinning and isobar spacing from the map span.
    - Larger span  => sparser barbs, slightly wider isobars
    - Smaller span => denser barbs, tighter isobars
    """
    w, e, s, n = extent
    lon_span = e - w               # already guaranteed e>w in centered frame
    lat_span = abs(n - s)
    span = max(lon_span, lat_span)

    # Heuristics tuned so CONUS stays detailed, WPAC/EPAC breathe
    if span >= 120:
        # very wide basin view (e.g., WPAC, global-ish)
        desired_x = 28            # smaller desired_x -> larger stride -> sparser
        barb_len = 5
        barb_min_stride = 9       # ensure sparsity
        mslp_lw = 0.95
        coast_lw = 0.9; border_lw = 0.75; state_lw = 0.5
        cint = 3                  # was 2; slightly wider spacing
    elif span >= 60:
        # large regional (EPAC/Atl basin chunks, wide continents)
        desired_x = 45
        barb_len = 6
        barb_min_stride = 8
        mslp_lw = 1.0
        coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6
        cint = 2
    elif span >= 30:
        # mesoscale-regional (EU, E/SE Asia subregions, etc.)
        desired_x = 65
        barb_len = 6
        barb_min_stride = 7
        mslp_lw = 1.05
        coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6
        cint = 3
    else:
        # tighter zooms (CONUS, states, small-country)
        desired_x = 85            # dense
        barb_len = 6
        barb_min_stride = 6
        mslp_lw = 1.1
        coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6
        cint = 4

    # convert desired density to strides; clamp for sanity
    stride_x = max(1, nx // desired_x)
    stride_y = max(1, ny // int(desired_x / 1.6))
    stride_x = max(stride_x, barb_min_stride)
    stride_y = max(stride_y, barb_min_stride)
    stride_x = min(stride_x, 14)
    stride_y = min(stride_y, 14)

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

# -------------------- Main renderer --------------------
def generate_visualization(year, month, day, hour, region_coords, api_key):
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"

    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": "reanalysis",
        "variable": [
            'mean_sea_level_pressure', '2m_dewpoint_temperature', '2m_temperature',
            '10m_u_component_of_wind', '10m_v_component_of_wind'
        ],
        "year": date_input[:4],
        "month": date_input[4:6],
        "day": date_input[6:8],
        "time": date_input[8:] + ":00",
        "format": "netcdf"
    }

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        target = tmp_file.name

    try:
        c.retrieve(dataset, request, target)
        ds = nc.Dataset(target)

        mslp = ds.variables['msl'][:] / 100.0
        d2m  = ds.variables['d2m'][:]
        u10  = ds.variables['u10'][:]
        v10  = ds.variables['v10'][:]
        lon0 = ds.variables['longitude'][:]
        lat  = ds.variables['latitude'][:]

        # enforce strictly increasing longitudes in 0..360
        order = np.argsort(lon0)
        lon_360 = lon0[order]
        mslp = mslp[:, :, order]
        d2m  = d2m[:,  :, order]
        u10  = u10[:,  :, order]
        v10  = v10[:,  :, order]

        lon_w, lon_e, lat_s, lat_n = region_coords

        # projection centered to the window midpoint (IDL-proof)
        proj, extent_crs, extent_centered, center_deg = build_projection_and_extent(lon_w, lon_e, lat_s, lat_n)

        # subset window (no cyclic padding needed)
        lon_sel, lat_sel, mslp_sub = subset_lon_lat(lon_360, lat, mslp, lon_w, lon_e, lat_s, lat_n)
        _,       _,       d2m_sub  = subset_lon_lat(lon_360, lat, d2m,  lon_w, lon_e, lat_s, lat_n)
        _,       _,       u10_sub  = subset_lon_lat(lon_360, lat, u10,  lon_w, lon_e, lat_s, lat_n)
        _,       _,       v10_sub  = subset_lon_lat(lon_360, lat, v10,  lon_w, lon_e, lat_s, lat_n)

        # ---- recenter selected longitudes to the map's center and then enforce monotonic axes
        lon_plot = to_center_frame_vec(lon_sel, center_deg)

        # tiny epsilon to avoid duplicate-edge issues exactly at +/-180
        lon_plot = np.where(np.isclose(np.diff(np.r_[lon_plot[0]-1e-8, lon_plot]), 0.0), lon_plot + 1e-8, lon_plot)

        lon_plot, lat_sel, mslp_sub, d2m_sub, u10_sub, v10_sub = ensure_increasing_axes(
            lon_plot, lat_sel, mslp_sub, d2m_sub, u10_sub, v10_sub
        )

        dewpoint_f = (d2m_sub - 273.15) * 9/5 + 32
        cmap = create_custom_colormap()
        date_str = read_valid_time(ds)

        data_crs_centered = ccrs.PlateCarree(central_longitude=center_deg)

        # figure/axes
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': proj})
        ax.set_extent(extent_centered, crs=extent_crs)

        params = auto_plot_params(extent_centered, nx=lon_plot.size, ny=lat_sel.size)

        # map features
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=params['border_lw'])
        try:
            ax.add_feature(cfeature.STATES, linestyle=':', linewidth=params['state_lw'])
        except Exception:
            pass

        # mesh for barbs
        LON2, LAT2 = np.meshgrid(lon_plot, lat_sel)

        # isobars
        mslp0 = np.ascontiguousarray(mslp_sub[0, :, :])
        cint = params['cint']
        mmin = np.floor(np.nanmin(mslp0) / cint) * cint
        mmax = np.ceil(np.nanmax(mslp0) / cint) * cint
        levels = np.arange(mmin, mmax + cint, cint)
        if levels.size > 60:
            skip = int(np.ceil(levels.size / 60))
            levels = levels[::skip]

        ax.contour(
            lon_plot, lat_sel, mslp0,
            levels=levels, colors='black',
            linewidths=params['mslp_lw'],
            transform=data_crs_centered
        )

        # filled dewpoint (now in the same centered frame)
        cf = ax.contourf(
            lon_plot, lat_sel, dewpoint_f[0, :, :],
            levels=np.linspace(-40, 90, 256), cmap=cmap, extend='both',
            transform=data_crs_centered
        )

        # wind barbs (zoom-aware thinning)
        si = params['stride_y']; sj = params['stride_x']
        ax.barbs(
            LON2[::si, ::sj],      LAT2[::si, ::sj],
            u10_sub[0, ::si, ::sj], v10_sub[0, ::si, ::sj],
            length=params['barb_len'],
            transform=data_crs_centered
        )

        # colorbar + title
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('2m Dewpoint Temperature (°F)')
        ax.set_title(f'ERA5 Pressure, Dewpoint, and Wind\nValid for: {date_str}\nPlotted by Sekai Chandra (@Sekai_WX)')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)

        ds.close()
        plt.close(fig)
        return buffer

    finally:
        if os.path.exists(target):
            os.remove(target)

# -------------------- Streamlit UI --------------------
st.title("ERA5 Weather Visualization")

# Get API key from secrets
try:
    api_key = st.secrets["CDS_API_KEY"]
except KeyError:
    st.error("CDS API key not found in secrets. Please configure your API key.")
    st.stop()

# inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.number_input("Year", min_value=1940, max_value=datetime.now().year, value=2023)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
with col4:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=12)

# region + button
col5, col6 = st.columns(2)
with col5:
    region_options = []
    for category, regions in REGIONS.items():
        for region_name in regions.keys():
            region_options.append(f"{category}: {region_name}")
    selected_region = st.selectbox("Select Region", region_options)
with col6:
    generate_button = st.button("Generate", type="primary", help="Generate the ERA5 visualization")

# run
if generate_button:
    category, region_name = selected_region.split(": ", 1)
    region_coords = REGIONS[category][region_name]
    try:
        with st.spinner("Downloading ERA5 data and generating visualization..."):
            image_buffer = generate_visualization(year, month, day, hour, region_coords, api_key)
        st.success("Visualization generated successfully!")
        st.image(image_buffer, caption=f"ERA5 Weather Data for {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC")
        st.download_button(
            label="Download Image",
            data=image_buffer,
            file_name=f"ERA5_{year}{month:02d}{day:02d}{hour:02d}_{region_name.replace(' ', '_')}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.info("Make sure the date/time is valid and the API service is available.")
