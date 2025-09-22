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
        "Northern Indian Ocean Basin": [-317, -256.3, -5, 34],
        "South Indian Ocean Basin": [32.7, 125.4, -44.8, 3.5],
        "Australian Basin": [100, 192.7, -50.2, -1.9]
    }
}

# -------------------- Color map --------------------
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

# -------------------- IDL-aware geo helpers --------------------
def crosses_dateline(lon_west, lon_east):
    """Check if a longitude range crosses the International Date Line."""
    return lon_west > lon_east

def normalize_longitude_range(lon_west, lon_east):
    """Normalize longitude range to handle IDL crossing."""
    # Convert to 0-360 range first
    lon_west = lon_west % 360
    lon_east = lon_east % 360
    
    if crosses_dateline(lon_west, lon_east):
        # For IDL crossing, we need to handle this specially
        return lon_west, lon_east, True
    else:
        return lon_west, lon_east, False

def get_projection_and_extent(region_coords):
    """Get appropriate projection and extent for the region."""
    lon_west, lon_east, lat_south, lat_north = region_coords
    
    # Normalize coordinates
    lon_west_norm, lon_east_norm, crosses_idl = normalize_longitude_range(lon_west, lon_east)
    
    if crosses_idl:
        # Use central longitude of 180 for IDL crossing
        central_lon = 180.0
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        
        # Convert extent to -180 to 180 range centered on 180
        extent_west = lon_west_norm - 360 if lon_west_norm > 180 else lon_west_norm
        extent_east = lon_east_norm if lon_east_norm <= 180 else lon_east_norm
        
        # Ensure west < east in the transformed coordinates
        if extent_west >= extent_east:
            extent_east += 360
            
        extent = [extent_west, extent_east, lat_south, lat_north]
    else:
        # Standard case - use central longitude in the middle of the region
        central_lon = (lon_west + lon_east) / 2
        if central_lon > 180:
            central_lon -= 360
        elif central_lon < -180:
            central_lon += 360
            
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        
        # Convert extent to appropriate range
        extent_west = lon_west if lon_west <= 180 else lon_west - 360
        extent_east = lon_east if lon_east <= 180 else lon_east - 360
        
        extent = [extent_west, extent_east, lat_south, lat_north]
    
    return proj, extent, crosses_idl

def get_data_subset(lon_data, lat_data, region_coords, data_arrays):
    """Get data subset for the region, handling IDL crossing."""
    lon_west, lon_east, lat_south, lat_north = region_coords
    lon_west_norm, lon_east_norm, crosses_idl = normalize_longitude_range(lon_west, lon_east)
    
    # Handle latitude (ensure proper order)
    lat_min, lat_max = min(lat_south, lat_north), max(lat_south, lat_north)
    lat_mask = (lat_data >= lat_min) & (lat_data <= lat_max)
    
    # Handle longitude
    if crosses_idl:
        # For IDL crossing, select points west of IDL OR east of IDL
        lon_mask = (lon_data >= lon_west_norm) | (lon_data <= lon_east_norm)
    else:
        # Standard case
        lon_mask = (lon_data >= lon_west_norm) & (lon_data <= lon_east_norm)
    
    # Apply masks
    lat_subset = lat_data[lat_mask]
    lon_subset = lon_data[lon_mask]
    
    # Subset data arrays
    data_subset = {}
    for name, data in data_arrays.items():
        if data.ndim == 3:  # time, lat, lon
            data_subset[name] = data[:, lat_mask, :][:, :, lon_mask]
        elif data.ndim == 2:  # lat, lon
            data_subset[name] = data[lat_mask, :][:, lon_mask]
    
    return lon_subset, lat_subset, data_subset

# -------------------- Auto-thinning (softer + denser isobars) --------------------
def auto_plot_params(extent, nx, ny):
    lon_span = abs(extent[1] - extent[0])
    lat_span = abs(extent[3] - extent[2])
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
    """Generate the ERA5 visualization."""
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"

    # CDS API
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

    # temp file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        target = tmp_file.name

    try:
        # retrieve & read
        c.retrieve(dataset, request, target)
        ds = nc.Dataset(target)

        # variables (ERA5 longitudes usually 0..360, latitude often descending)
        mslp = ds.variables['msl'][:] / 100.0  # hPa
        d2m  = ds.variables['d2m'][:]          # K
        u10  = ds.variables['u10'][:]
        v10  = ds.variables['v10'][:]
        lon_data = ds.variables['longitude'][:]
        lat_data = ds.variables['latitude'][:]

        # Ensure longitude is in 0-360 range and sorted
        lon_data = np.asarray(lon_data) % 360
        sort_idx = np.argsort(lon_data)
        lon_data = lon_data[sort_idx]
        
        # Sort all data arrays by longitude
        mslp = mslp[:, :, sort_idx]
        d2m = d2m[:, :, sort_idx]
        u10 = u10[:, :, sort_idx]
        v10 = v10[:, :, sort_idx]

        # Get appropriate projection and extent
        proj, extent, crosses_idl = get_projection_and_extent(region_coords)
        
        # Get data subset for the region
        data_arrays = {
            'mslp': mslp,
            'd2m': d2m,
            'u10': u10,
            'v10': v10
        }
        
        lon_subset, lat_subset, data_subset = get_data_subset(
            lon_data, lat_data, region_coords, data_arrays
        )

        # Convert longitude to plotting coordinates if needed
        if crosses_idl:
            # Convert to -180 to 180 range for plotting
            lon_plot = np.where(lon_subset > 180, lon_subset - 360, lon_subset)
            # Sort again if needed
            if not np.all(lon_plot[:-1] <= lon_plot[1:]):
                sort_idx = np.argsort(lon_plot)
                lon_plot = lon_plot[sort_idx]
                for key in data_subset:
                    if data_subset[key].ndim == 3:
                        data_subset[key] = data_subset[key][:, :, sort_idx]
                    elif data_subset[key].ndim == 2:
                        data_subset[key] = data_subset[key][:, sort_idx]
        else:
            lon_plot = lon_subset

        # Convert dewpoint to °F
        dewpoint_f = (data_subset['d2m'] - 273.15) * 9/5 + 32

        # colormap and title time
        cmap = create_custom_colormap()
        date_str = read_valid_time(ds)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': proj})
        ax.set_extent(extent, crs=proj)

        # Auto-thin based on subset size
        params = auto_plot_params(extent, nx=lon_plot.size, ny=lat_subset.size)

        # Map features
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=params['border_lw'])
        try:
            ax.add_feature(cfeature.STATES, linestyle=':', linewidth=params['state_lw'])
        except Exception:
            pass

        # Create meshgrid for plotting
        LON2, LAT2 = np.meshgrid(lon_plot, lat_subset)

        # Plot pressure contours
        mslp_plot = data_subset['mslp'][0, :, :]
        cint = params['cint']
        mmin = np.floor(np.nanmin(mslp_plot) / cint) * cint
        mmax = np.ceil(np.nanmax(mslp_plot) / cint) * cint
        levels = np.arange(mmin, mmax + cint, cint)
        if levels.size > 60:
            skip = int(np.ceil(levels.size / 60))
            levels = levels[::skip]

        ax.contour(
            LON2, LAT2, mslp_plot,
            levels=levels, colors='black',
            linewidths=params['mslp_lw'],
            transform=ccrs.PlateCarree()
        )

        # Plot filled dewpoint
        dp_plot = dewpoint_f[0, :, :]
        
        # Add cyclic point if needed for IDL crossing
        if crosses_idl and len(lon_plot) > 1:
            dp_cyc, lon_cyc = add_cyclic_point(dp_plot, coord=lon_plot)
            cf = ax.contourf(
                lon_cyc, lat_subset, dp_cyc,
                levels=np.linspace(-40, 90, 256), cmap=cmap, extend='both',
                transform=ccrs.PlateCarree()
            )
        else:
            cf = ax.contourf(
                LON2, LAT2, dp_plot,
                levels=np.linspace(-40, 90, 256), cmap=cmap, extend='both',
                transform=ccrs.PlateCarree()
            )

        # Wind barbs
        si = params['stride_y']
        sj = params['stride_x']
        ax.barbs(
            LON2[::si, ::sj], LAT2[::si, ::sj],
            data_subset['u10'][0, ::si, ::sj], data_subset['v10'][0, ::si, ::sj],
            length=params['barb_len'],
            transform=ccrs.PlateCarree()
        )

        # Colorbar and title
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.75)
        cb.set_label('2m Dewpoint Temperature (°F)')
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