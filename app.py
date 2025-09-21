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

# Page configuration
st.set_page_config(page_title="ERA5 Weather Visualization", layout="wide")

# Region coordinates dictionary (values can be outside [-180, 180]; we normalize)
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
        "Northern Indian Ocean Basin": [-317, -256.3, -5, 34],
        "South Indian Ocean Basin": [32.7, 125.4, -44.8, 3.5],
        "Australian Basin": [100, 192.7, -50.2, -1.9]
    }
}

def create_custom_colormap():
    """Custom colormap for dewpoint temperature."""
    colors = [(152, 109, 77), (150, 108, 76), (148, 107, 76), (146, 106, 75), (144, 105, 75), (142, 104, 74),
              (140, 102, 74), (138, 101, 73), (136, 100, 72), (134, 99, 72), (132, 98, 71), (130, 97, 71),
              (128, 96, 70), (126, 95, 70), (124, 94, 69), (122, 93, 68), (120, 91, 68), (118, 90, 67), (116, 89, 67),
              (114, 88, 66), (113, 87, 66), (111, 86, 65), (109, 85, 64), (107, 84, 64), (105, 83, 63), (103, 82, 63),
              (101, 80, 62), (99, 79, 61), (97, 78, 61), (95, 77, 60), (93, 76, 60), (91, 75, 59), (89, 74, 59),
              (87, 73, 58), (85, 72, 57), (83, 71, 57), (81, 69, 56), (79, 68, 56), (77, 67, 55), (75, 66, 55),
              (73, 65, 54), (71, 64, 54), (69, 63, 53), (77, 67, 52), (81, 71, 56), (86, 76, 60), (90, 80, 65),
              (94, 85, 69), (99, 89, 73), (103, 94, 77), (107, 98, 81), (112, 103, 86), (116, 107, 90), (120, 112, 94),
              (125, 116, 98), (129, 121, 103), (133, 125, 107), (138, 130, 111), (142, 134, 115), (146, 139, 119),
              (151, 143, 124), (155, 148, 128), (159, 152, 132), (164, 157, 137), (168, 161, 141), (173, 166, 145),
              (189, 179, 156), (189, 179, 156), (188, 184, 161), (193, 188, 165), (201, 197, 173), (201, 197, 173),
              (210, 206, 182), (223, 220, 194), (227, 224, 198), (231, 229, 202), (235, 233, 207), (240, 238, 211),
              (244, 242, 215), (230, 245, 230), (215, 240, 215), (200, 234, 200), (185, 229, 185), (170, 223, 170),
              (155, 218, 155), (140, 213, 140), (125, 207, 125), (110, 202, 110), (95, 196, 95), (80, 191, 80),
              (65, 186, 65), (48, 174, 48), (44, 163, 44), (39, 153, 39), (35, 142, 35), (30, 131, 30), (26, 121, 26),
              (21, 110, 21), (17, 99, 17), (12, 89, 12), (8, 78, 8), (97, 163, 175), (88, 150, 160), (80, 137, 146),
              (71, 123, 131), (62, 110, 116), (54, 97, 102), (45, 84, 87), (36, 70, 72), (28, 57, 58), (19, 44, 43),
              (102, 102, 154), (96, 94, 148), (89, 86, 142), (83, 78, 136), (77, 70, 130), (70, 62, 124), (64, 54, 118),
              (58, 46, 112), (51, 38, 106), (45, 30, 100), (114, 64, 113), (120, 69, 115), (125, 75, 117),
              (131, 80, 118), (136, 86, 120), (142, 91, 122), (147, 97, 124), (153, 102, 125), (158, 108, 127),
              (164, 113, 129)]
    norm_colors = [(r/255, g/255, b/255) for r, g, b in colors]
    return mcolors.LinearSegmentedColormap.from_list('custom_dewpoint', norm_colors, N=256)

# -------------------- Geo helpers --------------------

def normalize_extent(ext):
    """Normalize [lon_min, lon_max, lat_min, lat_max] to lon in [-180, 180)."""
    lon0, lon1, lat0, lat1 = ext
    def norm_lon(x):
        return ((x + 180) % 360) - 180
    return [norm_lon(lon0), norm_lon(lon1), lat0, lat1]

def ensure_lat_order(ext):
    """Guarantee lat_min < lat_max to avoid blank maps when extents are inverted."""
    lon0, lon1, lat0, lat1 = ext
    if lat0 > lat1:
        lat0, lat1 = lat1, lat0
    return [lon0, lon1, lat0, lat1]

def crosses_dateline(ext_norm):
    """True if the normalized extent crosses the dateline (i.e., west > east)."""
    lon0, lon1, _, _ = ext_norm
    return lon0 > lon1

def to180(lon):
    """Shift longitude into the 180-centered frame [-180,180)."""
    return ((lon - 180 + 180) % 360) - 180

def extent_to_180(ext_norm, eps=1e-6):
    """
    Convert normalized extent (which crosses the dateline) into the 180-centered frame.
    Add a tiny epsilon away from ±180 to avoid seam ambiguity.
    """
    lon0, lon1, lat0, lat1 = ext_norm
    w = to180(lon0)
    e = to180(lon1)
    # After recentering, a crossing extent should become non-crossing
    if w > e:
        w, e = e, w
    # nudge off the seam
    if abs(w + 180) < 1e-5: w += eps
    if abs(e - 180) < 1e-5: e -= eps
    return [w, e, lat0, lat1]

# -------------------- Auto-thinning (softer, denser isobars) --------------------

def auto_plot_params(normed_extent, nx, ny):
    """
    Softer thinning: keep more detail at all scales.
    Returns dict with barb/line density and isobar spacing.
    """
    lon0, lon1, lat0, lat1 = normed_extent
    lon_span = (lon1 - lon0) % 360.0
    if lon_span > 180:
        lon_span = 360 - lon_span
    lat_span = abs(lat1 - lat0)
    span = max(lon_span, lat_span)

    # Denser isobars
    if span >= 120:         # basin/hemisphere
        desired_x = 40
        barb_len  = 5
        mslp_lw   = 1.0
        coast_lw  = 0.9
        border_lw = 0.7
        state_lw  = 0.5
        cint      = 2       # denser than before
    elif span >= 60:        # sub-basin / multi-country
        desired_x = 55
        barb_len  = 6
        mslp_lw   = 1.1
        coast_lw  = 1.0
        border_lw = 0.8
        state_lw  = 0.6
        cint      = 2
    elif span >= 30:        # large region
        desired_x = 70
        barb_len  = 6
        mslp_lw   = 1.1
        coast_lw  = 1.0
        border_lw = 0.8
        state_lw  = 0.6
        cint      = 2
    else:                   # zoomed-in
        desired_x = 85
        barb_len  = 7
        mslp_lw   = 1.2
        coast_lw  = 1.0
        border_lw = 0.8
        state_lw  = 0.6
        cint      = 2

    stride_x = max(1, nx // desired_x)
    stride_y = max(1, ny // int(desired_x / 1.6))
    stride_x = min(stride_x, 8 if span < 150 else 12)
    stride_y = min(stride_y, 8 if span < 150 else 12)

    return {
        'stride_y': stride_y,
        'stride_x': stride_x,
        'barb_len': barb_len,
        'mslp_lw': mslp_lw,
        'coast_lw': coast_lw,
        'border_lw': border_lw,
        'state_lw': state_lw,
        'cint': cint
    }

# -------------------- Time label helper --------------------

def read_valid_time(ds):
    """Read a nice datetime string from ERA5 single-level netCDF."""
    var = ds.variables.get('valid_time') or ds.variables.get('time')
    if var is None:
        return "Unknown valid time"
    time_unit = getattr(var, 'units', None)
    time_calendar = getattr(var, 'calendar', 'standard')
    tvals = var[:]
    try:
        date = nc.num2date(tvals[0], units=time_unit, calendar=time_calendar)
        return date.strftime("%B %d, %Y - %H:%M UTC")
    except Exception:
        return "Unknown valid time"

# -------------------- Main renderer --------------------

def generate_visualization(year, month, day, hour, region_coords, api_key):
    """Generate the ERA5 visualization."""
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"

    # Configure CDS API
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)

    # ERA5 single-level request
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

    # Temp file for download
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        target = tmp_file.name

    try:
        # Retrieve data
        c.retrieve(dataset, request, target)

        # Load netCDF
        ds = nc.Dataset(target)

        # Variables
        mslp = ds.variables['msl'][:] / 100.0  # hPa
        d2m  = ds.variables['d2m'][:]          # K
        u10  = ds.variables['u10'][:]
        v10  = ds.variables['v10'][:]
        lon0_360 = ds.variables['longitude'][:]   # typically 0..360
        lat = ds.variables['latitude'][:]        # often descending

        # Time label
        date_str = read_valid_time(ds)

        # Build longitude domains
        lon_360 = np.asarray(lon0_360)
        order_360 = np.argsort(lon_360)
        lon_360 = lon_360[order_360]
        mslp_360 = mslp[:, :, order_360]
        d2m_360  = d2m[:, :, order_360]
        u10_360  = u10[:, :, order_360]
        v10_360  = v10[:, :, order_360]

        lon_m180 = np.where(lon_360 > 180, lon_360 - 360, lon_360)
        order_m180 = np.argsort(lon_m180)
        lon_m180 = lon_m180[order_m180]
        mslp_m180 = mslp_360[:, :, order_m180]
        d2m_m180  = d2m_360[:, :, order_m180]
        u10_m180  = u10_360[:, :, order_m180]
        v10_m180  = v10_360[:, :, order_m180]

        # Normalize & sanitize extent
        normed_extent = ensure_lat_order(normalize_extent(region_coords))
        crossing = crosses_dateline(normed_extent)

        # Choose projection/coords cleanly
        if crossing:
            # 180-centered projection + data/extent in that frame
            proj = ccrs.PlateCarree(central_longitude=180)
            data_crs = ccrs.PlateCarree(central_longitude=180)

            # Shift lon to 180-centered frame and add cyclic point to avoid seam gap
            lon_180 = np.where(lon_360 >= 180, lon_360 - 360, lon_360)  # [-180,180)
            lon_180_sorted = np.sort(lon_180)
            # reorder all fields to match lon_180_sorted
            idx = np.argsort(lon_180)
            lon_180_sorted = lon_180[idx]
            mslp_180 = mslp_360[:, :, idx]
            d2m_180  = d2m_360[:, :, idx]
            u10_180  = u10_360[:, :, idx]
            v10_180  = v10_360[:, :, idx]

            # add cyclic in lon for contourf/contour stability at seam
            mslp_180_c, lon_180_c = add_cyclic_point(mslp_180[0, :, :], coord=lon_180_sorted)
            d2m_180_c,  _         = add_cyclic_point(d2m_180[0, :, :],  coord=lon_180_sorted)
            u10_180_c,  _         = add_cyclic_point(u10_180[0, :, :],  coord=lon_180_sorted)
            v10_180_c,  _         = add_cyclic_point(v10_180[0, :, :],  coord=lon_180_sorted)

            # extent in 180-centered frame with tiny epsilon off the seam
            plot_extent = ensure_lat_order(extent_to_180(normed_extent, eps=1e-4))

            # plotting lon/fields
            plot_lon_1d = lon_180_c
            mslp0 = mslp_180_c
            d2m0  = d2m_180_c
            u10_0 = u10_180_c
            v10_0 = v10_180_c

        else:
            # default 0-centered projection + [-180,180) data
            proj = ccrs.PlateCarree()
            data_crs = ccrs.PlateCarree()
            plot_extent = normed_extent

            plot_lon_1d = lon_m180
            mslp0 = mslp_m180[0, :, :]
            d2m0  = d2m_m180[0, :, :]
            u10_0 = u10_m180[0, :, :]
            v10_0 = v10_m180[0, :, :]

        # Convert dewpoint to °F
        dewpoint_f0 = (d2m0 - 273.15) * 9/5 + 32

        # Colormap
        cmap = create_custom_colormap()

        # Figure/Axes
        fig, ax = plt.subplots(figsize=(18, 9), subplot_kw={'projection': proj})
        ax.set_extent(plot_extent, crs=proj)  # extent expressed in the same CRS as projection

        # Auto-thinning + denser isobars
        params = auto_plot_params(normed_extent, nx=plot_lon_1d.size, ny=lat.size)

        # Map features (drop STATES for global basins to avoid heavy draw)
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=params['border_lw'])
        if not crossing:
            ax.add_feature(cfeature.STATES, linestyle=':', linewidth=params['state_lw'])

        # 2D coords for barbs
        LON2, LAT2 = np.meshgrid(plot_lon_1d, lat)

        # Isobars (denser, with safety cap on level count)
        cint = params['cint']
        mmin = np.floor(np.nanmin(mslp0) / cint) * cint
        mmax = np.ceil(np.nanmax(mslp0) / cint) * cint
        levels = np.arange(mmin, mmax + cint, cint)
        if levels.size > 80:
            levels = levels[::2]  # thin the list, not the value spacing, to keep speed

        cs = ax.contour(
            plot_lon_1d, lat, mslp0,
            levels=levels, colors='black',
            linewidths=params['mslp_lw'],
            transform=data_crs
        )

        # Filled dewpoint
        dp_levels = np.linspace(-40, 90, 256)
        cf = ax.contourf(
            plot_lon_1d, lat, dewpoint_f0,
            levels=dp_levels, cmap=cmap, extend='both',
            transform=data_crs
        )

        # Wind barbs (thinned)
        si = params['stride_y']
        sj = params['stride_x']
        ax.barbs(
            LON2[::si, ::sj], LAT2[::si, ::sj],
            u10_0[::si, ::sj], v10_0[::si, ::sj],
            length=params['barb_len'],
            transform=data_crs
        )

        # Colorbar
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.7)
        cb.set_label('2m Dewpoint Temperature (°F)')

        # Title
        ax.set_title(f'ERA5 Pressure, Dewpoint, and Wind\nValid for: {date_str}\nPlotted by Sekai Chandra (@Sekai_WX)')

        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)

        # Cleanup
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

# Date/time inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.number_input("Year", min_value=1940, max_value=datetime.now().year, value=2023)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
with col4:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=12)

# Region & generate
col5, col6 = st.columns(2)
with col5:
    region_options = []
    for category, regions in REGIONS.items():
        for region_name in regions.keys():
            region_options.append(f"{category}: {region_name}")
    selected_region = st.selectbox("Select Region", region_options)
with col6:
    generate_button = st.button("Generate", type="primary",
                                help="Generate the ERA5 visualization")

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
