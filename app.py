import streamlit as st
import cdsapi
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import os
import tempfile
from datetime import datetime
import io

# -------------------- Streamlit page --------------------
st.set_page_config(page_title="ERA5 Weather Visualization", layout="wide")

# Region coordinates dictionary (values can be outside [-180, 180]; we normalize for viewing)
REGIONS = {
    "General": {
        "Continental United States": [-125, -66.5, 24.396308, 49.384358],
        "North Atlantic Basin": [-102.8, -7.9, 6, 57.6],
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
        "West Pacific Basin": [94.9, 183.5, -14.6, 56.1],       # crosses dateline
        "East Pacific Basin": [-161.4, -86.3, 3, 39],
        "Central Pacific Basin": [-188.8, -141.6, 2.4, 41.1],   # crosses dateline
        "Northern Indian Ocean Basin": [-317, -256.3, -5, 34],  # crosses dateline in raw numbers
        "South Indian Ocean Basin": [32.7, 125.4, -44.8, 3.5],
        "Australian Basin": [100, 192.7, -50.2, -1.9]           # crosses dateline
    }
}

# -------------------- Color map --------------------
def create_custom_colormap():
    """Custom colormap for dewpoint temperature."""
    colors = [
        (152,109,77),(150,108,76),(148,107,76),(146,106,75),(144,105,75),(142,104,74),
        (140,102,74),(138,101,73),(136,100,72),(134,99,72),(132,98,71),(130,97,71),
        (128,96,70),(126,95,70),(124,94,69),(122,93,68),(120,91,68),(118,90,67),
        (116,89,67),(114,88,66),(113,87,66),(109,85,64),(107,84,64),(105,83,63),
        (103,82,63),(101,80,62),(99,79,61),(97,78,61),(95,77,60),(93,76,60),
        (91,75,59),(89,74,59),(87,73,58),(85,72,57),(83,71,57),(81,69,56),
        (79,68,56),(77,67,55),(75,66,55),(73,65,54),(71,64,54),(69,63,53),
        (77,67,52),(81,71,56),(86,76,60),(90,80,65),(94,85,69),(99,89,73),
        (103,94,77),(107,98,81),(112,103,86),(116,107,90),(120,112,94),(125,116,98),
        (129,121,103),(133,125,107),(138,130,111),(142,134,115),(146,139,119),
        (151,143,124),(155,148,128),(159,152,132),(164,157,137),(168,161,141),
        (173,166,145),(189,179,156),(189,179,156),(188,184,161),(193,188,165),
        (201,197,173),(201,197,173),(210,206,182),(223,220,194),(227,224,198),
        (231,229,202),(235,233,207),(240,238,211),(244,242,215),(230,245,230),
        (215,240,215),(200,234,200),(185,229,185),(170,223,170),(155,218,155),
        (140,213,140),(125,207,125),(110,202,110),(95,196,95),(80,191,80),
        (65,186,65),(48,174,48),(44,163,44),(39,153,39),(35,142,35),(30,131,30),
        (26,121,26),(21,110,21),(17,99,17),(12,89,12),(8,78,8),(97,163,175),
        (88,150,160),(80,137,146),(71,123,131),(62,110,116),(54,97,102),(45,84,87),
        (36,70,72),(28,57,58),(19,44,43),(102,102,154),(96,94,148),(89,86,142),
        (83,78,136),(77,70,130),(70,62,124),(64,54,118),(58,46,112),(51,38,106),
        (45,30,100),(114,64,113),(120,69,115),(125,75,117),(131,80,118),(136,86,120),
        (142,91,122),(147,97,124),(153,102,125),(158,108,127),(164,113,129)
    ]
    norm_colors = [(r/255, g/255, b/255) for r, g, b in colors]
    return mcolors.LinearSegmentedColormap.from_list('custom_dewpoint', norm_colors, N=256)

# -------------------- Geo helpers (IDL-safe) --------------------
def norm_pm180(lon):
    """Normalize any lon to (-180, 180]; keep exact -180 at +180."""
    x = (lon + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(x, -180.0) else x

def to_pm180(lon):
    return norm_pm180(lon)

def crosses_dateline_box(lon_w, lon_e):
    """Return True if [lon_w, lon_e] crosses the IDL in a -180..180 frame."""
    w = norm_pm180(lon_w); e = norm_pm180(lon_e)
    return e <= w  # example: 170E..-150W

def extent_for_central180(ext):
    """
    Convert [lon_w, lon_e, lat_s, lat_n] (any range; may cross IDL) into an extent
    suitable for PlateCarree(central_longitude=180) where west < east.
    If the box crosses the IDL, we let east exceed +180 by adding 360.
    """
    lon_w, lon_e, lat_s, lat_n = ext
    w = to_pm180(lon_w)
    e = to_pm180(lon_e)
    if lat_s > lat_n:
        lat_s, lat_n = lat_n, lat_s
    if e <= w:
        e += 360.0
    return [w, e, lat_s, lat_n]

def span_from_extent180(ext180):
    w, e, _, _ = ext180
    return (e - w) if e >= w else (e + 360.0 - w)

def lon360_in_extent180_mask(lon360, ext180):
    """
    Mask longitudes in [0,360) that fall inside an extent defined for
    PlateCarree(central_longitude=180). Handles IDL crossing by unwrapping.
    """
    w, e, _, _ = ext180
    lon_c180 = ((lon360 + 180.0) % 360.0) - 180.0
    if e > 180.0:
        lon_unwrapped = np.where(lon_c180 < w, lon_c180 + 360.0, lon_c180)
    else:
        lon_unwrapped = lon_c180
    return (lon_unwrapped >= w) & (lon_unwrapped <= e)

def lat_mask_for_extent(lat, ext180):
    """Mask latitude array (which may be descending) to [lat_s, lat_n]."""
    _, _, s, n = ext180
    lo, hi = min(s, n), max(s, n)
    return (lat >= lo) & (lat <= hi) if lat[0] < lat[-1] else (lat <= hi) & (lat >= lo)

# -------------------- CDS area helpers (subset + stitch) --------------------
def areas_for_cds(ext):
    """
    Convert [lon_w, lon_e, lat_s, lat_n] to one or two CDS 'area' boxes [N,W,S,E] in -180..180.
    If the request crosses the IDL, return two boxes split at 180.
    """
    lon_w, lon_e, lat_s, lat_n = ext
    north, south = max(lat_s, lat_n), min(lat_s, lat_n)
    w = norm_pm180(lon_w)
    e = norm_pm180(lon_e)

    if not crosses_dateline_box(lon_w, lon_e):
        west, east = min(w, e), max(w, e)
        return [[north, west, south, east]]
    else:
        # Split into [w..180] and [-180..e]
        west1, east1 = (w, 180.0)
        west2, east2 = (-180.0, e)
        return [
            [north, west1, south, east1],
            [north, west2, south, east2],
        ]

def concat_lon_slices(lon_list, arr_list):
    """
    Concatenate multiple longitude slices (same lat/time dims) and return sorted by lon.
    arr_list elements are shaped (time, lat, lon).
    """
    lon_all = np.concatenate(lon_list)
    data_all = np.concatenate(arr_list, axis=-1)
    order = np.argsort(lon_all)
    lon_sorted = lon_all[order]
    data_sorted = data_all[..., order]
    return lon_sorted, data_sorted

# -------------------- Auto-thinning (softer + denser isobars) --------------------
def auto_plot_params(ext180, nx, ny):
    lon_span = span_from_extent180(ext180)
    lat_span = abs(ext180[3] - ext180[2])
    span = max(lon_span, lat_span)

    if span >= 120:         # basin/hemisphere
        desired_x = 40; barb_len = 5;  mslp_lw = 0.9; coast_lw = 0.9; border_lw = 0.7; state_lw = 0.5; cint = 2
    elif span >= 60:        # sub-basin
        desired_x = 55; barb_len = 6;  mslp_lw = 1.0; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 2
    elif span >= 30:        # large region
        desired_x = 70; barb_len = 6;  mslp_lw = 1.1; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 2
    else:                   # zoomed-in
        desired_x = 85; barb_len = 7;  mslp_lw = 1.2; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 2

    stride_x = max(1, nx // desired_x)
    stride_y = max(1, ny // int(desired_x / 1.6))
    stride_x = min(stride_x, 8 if span < 150 else 12)
    stride_y = min(stride_y, 8 if span < 150 else 12)

    return {'stride_y': stride_y, 'stride_x': stride_x, 'barb_len': barb_len,
            'mslp_lw': mslp_lw, 'coast_lw': coast_lw, 'border_lw': border_lw,
            'state_lw': state_lw, 'cint': cint}

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
    """Generate the ERA5 visualization (CDS-subset + IDL robust)."""
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"

    # CDS API client
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)
    dataset = "reanalysis-era5-single-levels"

    # Convert UI region to one or two CDS 'area' boxes (N,W,S,E) in -180..180
    cds_boxes = areas_for_cds(region_coords)

    tmp_paths = []
    try:
        lon_chunks, mslp_chunks, d2m_chunks, u10_chunks, v10_chunks = [], [], [], [], []
        lat_ref = None
        date_str = None

        for box in cds_boxes:
            area_n, area_w, area_s, area_e = box
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
                "area": [area_n, area_w, area_s, area_e],  # N, W, S, E  (W/E in -180..180)
                "format": "netcdf"
            }

            # temp file per box
            tmpf = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
            tmp_paths.append(tmpf.name)
            tmpf.close()

            c.retrieve(dataset, request, tmp_paths[-1])

            # read and collect
            ds = nc.Dataset(tmp_paths[-1])

            if date_str is None:
                date_str = read_valid_time(ds)

            # variables
            mslp = ds.variables['msl'][:] / 100.0  # hPa
            d2m  = ds.variables['d2m'][:]          # K
            u10  = ds.variables['u10'][:]
            v10  = ds.variables['v10'][:]
            lat  = ds.variables['latitude'][:]
            lon  = ds.variables['longitude'][:]

            # ERA5 longitudes are usually 0..360; enforce 0..360 for stitching
            lon_360 = (np.asarray(lon) % 360.0)

            # Sort by lon ascending for this chunk
            order = np.argsort(lon_360)
            lon_360 = lon_360[order]
            mslp = mslp[:, :, order]
            d2m  = d2m[:,  :, order]
            u10  = u10[:,  :, order]
            v10  = v10[:,  :, order]

            if lat_ref is None:
                lat_ref = lat
            else:
                # sanity: lats must match
                if lat.shape != lat_ref.shape or not np.allclose(lat, lat_ref):
                    ds.close()
                    raise RuntimeError("Latitude arrays differ between split CDS tiles; cannot stitch.")

            lon_chunks.append(lon_360)
            mslp_chunks.append(mslp)
            d2m_chunks.append(d2m)
            u10_chunks.append(u10)
            v10_chunks.append(v10)
            ds.close()

        # Stitch across longitude if there were two parts
        lon_all, mslp_all = concat_lon_slices(lon_chunks, mslp_chunks)
        _,       d2m_all  = concat_lon_slices(lon_chunks, d2m_chunks)
        _,       u10_all  = concat_lon_slices(lon_chunks, u10_chunks)
        _,       v10_all  = concat_lon_slices(lon_chunks, v10_chunks)

        # Build extent for a dateline-centered view (for map set_extent)
        extent180 = extent_for_central180(region_coords)

        # Because we already downloaded only the requested box, we can skip extra subsetting.
        lon_sub = lon_all
        lat_sub = lat_ref

        # convert dewpoint to °F
        dewpoint_f = (d2m_all - 273.15) * 9/5 + 32

        # ---- Projection setup (IDL-safe) ----
        proj = ccrs.PlateCarree(central_longitude=180)
        extent_crs = ccrs.PlateCarree(central_longitude=180)
        data_crs = ccrs.PlateCarree()  # data lon are 0..360, ok

        # figure/axes
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': proj})
        ax.set_extent(extent180, crs=extent_crs)

        # auto-thin based on subset size
        params = auto_plot_params(extent180, nx=lon_sub.size, ny=lat_sub.size)

        # map features
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=params['border_lw'])
        try:
            ax.add_feature(cfeature.STATES, linestyle=':', linewidth=params['state_lw'])
        except Exception:
            pass

        # 2D grid for barbs
        LON2, LAT2 = np.meshgrid(lon_sub, lat_sub)

        # Determine if subset crosses/touches the 0/360 seam
        touches_seam = crosses_dateline_box(region_coords[0], region_coords[1])

        # ----- Filled dewpoint (use cyclic when touching seam) -----
        dp_slice = dewpoint_f[0, :, :]
        lon_for_fill = lon_sub
        if touches_seam:
            dp_slice, lon_for_fill = add_cyclic_point(dp_slice, coord=lon_for_fill)

        cf = ax.contourf(
            lon_for_fill, lat_sub, dp_slice,
            levels=np.linspace(-40, 90, 256),
            cmap=create_custom_colormap(), extend='both',
            transform=data_crs
        )

        # ----- Isobars (also cyclic when touching seam) -----
        mslp0 = mslp_all[0, :, :]
        mmin = np.floor(np.nanmin(mslp0) / params['cint']) * params['cint']
        mmax = np.ceil(np.nanmax(mslp0) / params['cint']) * params['cint']
        levels = np.arange(mmin, mmax + params['cint'], params['cint'])
        if levels.size > 60:
            skip = int(np.ceil(levels.size / 60))
            levels = levels[::skip]

        mslp_field = mslp0
        lon_for_contour = lon_sub
        if touches_seam:
            mslp_field, lon_for_contour = add_cyclic_point(mslp_field, coord=lon_for_contour)

        ax.contour(
            lon_for_contour, lat_sub, mslp_field,
            levels=levels, colors='black',
            linewidths=params['mslp_lw'],
            transform=data_crs
        )

        # wind barbs (thinned by subset-based strides)
        si = params['stride_y']; sj = params['stride_x']
        ax.barbs(
            LON2[::si, ::sj], LAT2[::si, ::sj],
            u10_all[0, ::si, ::sj], v10_all[0, ::si, ::sj],
            length=params['barb_len'],
            transform=data_crs
        )

        # colorbar and title
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('2m Dewpoint Temperature (°F)')
        ax.set_title(f'ERA5 Pressure, Dewpoint, and Wind\nValid for: {date_str}\nPlotted by Sekai Chandra (@Sekai_WX)')

        # save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)

        plt.close(fig)
        return buffer

    finally:
        for p in tmp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

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
