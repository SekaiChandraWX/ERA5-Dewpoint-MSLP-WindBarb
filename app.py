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
    """
    Normalize [lon_min, lon_max, lat_min, lat_max] to lon in [-180, 180).
    Works even if the input longitudes are outside that range.
    """
    lon0, lon1, lat0, lat1 = ext
    def norm_lon(x):
        return ((x + 180) % 360) - 180
    return [norm_lon(lon0), norm_lon(lon1), lat0, lat1]

def extent_span_deg(ext_norm):
    """Return (lon_span_deg, lat_span_deg, span_proxy) for a normalized extent (wrap-aware)."""
    lon0, lon1, lat0, lat1 = ext_norm
    lon_span = (lon1 - lon0) % 360.0
    if lon_span > 180:
        lon_span = 360 - lon_span
    lat_span = abs(lat1 - lat0)
    return lon_span, lat_span, max(lon_span, lat_span)

def choose_projection_and_extent(ext_norm):
    """
    Dateline-aware choice:
      - If lon0 > lon1 (crosses the dateline), use PlateCarree(central_longitude=180)
        and provide extent in that CRS.
      - Else use default PlateCarree.
    """
    lon0, lon1, lat0, lat1 = ext_norm
    crosses_dateline = lon0 > lon1
    if crosses_dateline:
        proj = ccrs.PlateCarree(central_longitude=180)
        # convert extent longitudes into the 180-centered frame
        def to180(lon):
            return ((lon - 180 + 360) % 360) - 180
        extent_180 = [to180(lon0), to180(lon1), lat0, lat1]
        extent_crs = ccrs.PlateCarree(central_longitude=180)
        return proj, extent_180, extent_crs
    else:
        proj = ccrs.PlateCarree()
        extent_crs = ccrs.PlateCarree()
        return proj, ext_norm, extent_crs

# -------------------- Auto-thinning (softer) --------------------

def auto_plot_params(normed_extent, nx, ny):
    """
    Softer thinning: keep more detail at all scales.
    Returns:
      { 'stride_y', 'stride_x', 'barb_len', 'mslp_lw', 'coast_lw', 'border_lw', 'state_lw', 'cint' }
    """
    lon_span, lat_span, span = extent_span_deg(normed_extent)

    if span >= 120:         # basin/hemisphere
        desired_x = 40      # denser than before
        barb_len  = 5
        mslp_lw   = 0.8
        coast_lw  = 0.8
        border_lw = 0.6
        state_lw  = 0.4
        cint      = 4
    elif span >= 60:        # sub-basin / multi-country
        desired_x = 55
        barb_len  = 6
        mslp_lw   = 0.9
        coast_lw  = 0.9
        border_lw = 0.7
        state_lw  = 0.5
        cint      = 3
    elif span >= 30:        # large country / region
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

    # convert desired_x to strides using available grid size
    stride_x = max(1, nx // desired_x)
    stride_y = max(1, ny // int(desired_x / 1.6))

    # safety: never thin beyond every 8th gp (or 12th if near-global)
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
        d2m = ds.variables['d2m'][:]           # K
        t2m = ds.variables['t2m'][:]           # K
        u10 = ds.variables['u10'][:]
        v10 = ds.variables['v10'][:]
        lon_raw = ds.variables['longitude'][:]  # usually 0..360
        lat_raw = ds.variables['latitude'][:]   # often descending

        # Time label
        date_str = read_valid_time(ds)

        # Longitude wrap to [-180, 180) and reorder to increasing
        lon = np.where(lon_raw > 180, lon_raw - 360, lon_raw)
        lon_order = np.argsort(lon)
        lon = lon[lon_order]
        mslp = mslp[:, :, lon_order]
        d2m = d2m[:, :, lon_order]
        t2m = t2m[:, :, lon_order]
        u10 = u10[:, :, lon_order]
        v10 = v10[:, :, lon_order]

        # Latitude as-is
        lat = lat_raw

        # Convert temps to °F for plotting (dewpoint)
        dewpoint = (d2m - 273.15) * 9/5 + 32

        # Colormap
        cmap = create_custom_colormap()

        # Normalize region and pick projection/extent (dateline-aware)
        normed_extent = normalize_extent(region_coords)
        proj, plot_extent, extent_crs = choose_projection_and_extent(normed_extent)

        # Figure/Axes
        fig, ax = plt.subplots(figsize=(18, 9), subplot_kw={'projection': proj})
        ax.set_extent(plot_extent, crs=extent_crs)

        # Auto-thinning + line scaling
        params = auto_plot_params(normed_extent, nx=lon.size, ny=lat.size)

        # Map features
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=params['border_lw'])
        ax.add_feature(cfeature.STATES, linestyle=':', linewidth=params['state_lw'])

        # Build 2D coords for barbs
        LON2, LAT2 = np.meshgrid(lon, lat)

        # Isobars
        mslp0 = mslp[0, :, :]
        cint = params['cint']
        mmin = np.floor(mslp0.min() / cint) * cint
        mmax = np.ceil(mslp0.max() / cint) * cint
        mlevels = np.arange(mmin, mmax + cint, cint)

        cs = ax.contour(
            lon, lat, mslp0,
            levels=mlevels,
            colors='black',
            linewidths=params['mslp_lw'],
            transform=ccrs.PlateCarree()
        )

        # Filled dewpoint
        levels = np.linspace(-40, 90, 256)
        cf = ax.contourf(
            lon, lat, dewpoint[0, :, :],
            levels=levels, cmap=cmap, extend='both',
            transform=ccrs.PlateCarree()
        )

        # Wind barbs (thinned)
        si = params['stride_y']
        sj = params['stride_x']
        ax.barbs(
            LON2[::si, ::sj], LAT2[::si, ::sj],
            u10[0, ::si, ::sj], v10[0, ::si, ::sj],
            length=params['barb_len'],
            transform=ccrs.PlateCarree()
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
