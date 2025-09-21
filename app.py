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

# -------------------- Coordinate helpers --------------------
def normalize_longitude(lon):
    """Normalize longitude to [0, 360) range."""
    return lon % 360.0

def to_pm180(lon):
    """Convert longitude to (-180, 180] range."""
    x = (lon + 180.0) % 360.0 - 180.0
    return 180.0 if np.isclose(x, -180.0) else x

def crosses_dateline(lon_w, lon_e):
    """Check if a longitude range crosses the dateline."""
    # Normalize to 0-360 for comparison
    w_norm = normalize_longitude(lon_w)
    e_norm = normalize_longitude(lon_e)
    return e_norm < w_norm

def region_to_cds_area(region_coords, buffer_deg=5.0):
    """
    Convert region coordinates to CDS area format with buffer.
    Returns area string(s) for CDS API request.
    CDS area format: "north/west/south/east"
    """
    lon_w, lon_e, lat_s, lat_n = region_coords
    
    # Add buffer
    lat_s = max(-90, lat_s - buffer_deg)
    lat_n = min(90, lat_n + buffer_deg)
    lon_w = lon_w - buffer_deg
    lon_e = lon_e + buffer_deg
    
    # Ensure lat_s < lat_n
    if lat_s > lat_n:
        lat_s, lat_n = lat_n, lat_s
    
    if crosses_dateline(lon_w, lon_e):
        # Split into two areas for IDL crossing
        area1 = f"{lat_n}/{normalize_longitude(lon_w)}/{lat_s}/360"
        area2 = f"{lat_n}/0/{lat_s}/{normalize_longitude(lon_e)}"
        return [area1, area2]
    else:
        # Single area
        w_norm = normalize_longitude(lon_w)
        e_norm = normalize_longitude(lon_e)
        if e_norm < w_norm:
            e_norm += 360
        return [f"{lat_n}/{w_norm}/{lat_s}/{e_norm}"]

def merge_datasets(ds_list):
    """Merge multiple netCDF datasets along longitude dimension."""
    if len(ds_list) == 1:
        return ds_list[0]
    
    # Read data from all datasets
    all_data = {}
    all_lons = []
    lat = None
    
    for i, ds in enumerate(ds_list):
        lon = ds.variables['longitude'][:]
        if lat is None:
            lat = ds.variables['latitude'][:]
        
        all_lons.append(lon)
        
        # Store data for each variable
        for var_name in ['msl', 'd2m', 'u10', 'v10']:
            if var_name not in all_data:
                all_data[var_name] = []
            all_data[var_name].append(ds.variables[var_name][:])
    
    # Concatenate longitudes and sort
    combined_lon = np.concatenate(all_lons)
    sort_idx = np.argsort(combined_lon)
    final_lon = combined_lon[sort_idx]
    
    # Concatenate and sort data
    final_data = {}
    for var_name in all_data:
        combined_data = np.concatenate(all_data[var_name], axis=2)
        final_data[var_name] = combined_data[:, :, sort_idx]
    
    return final_lon, lat, final_data

# -------------------- Projection helpers --------------------
def get_projection_setup(region_coords):
    """Determine the best projection setup for the region."""
    lon_w, lon_e, lat_s, lat_n = region_coords
    
    if crosses_dateline(lon_w, lon_e):
        # Use dateline-centered projection for IDL crossing regions
        central_lon = 180.0
    else:
        # Use standard projection
        central_lon = 0.0
    
    return central_lon

def prepare_plot_extent(region_coords, central_lon):
    """Prepare the plot extent for the given central longitude."""
    lon_w, lon_e, lat_s, lat_n = region_coords
    
    if central_lon == 180.0:
        # Convert to -180 to 180 range centered on dateline
        w = to_pm180(lon_w)
        e = to_pm180(lon_e)
        if e <= w:
            e += 360.0
    else:
        # Use 0-360 range
        w = normalize_longitude(lon_w)
        e = normalize_longitude(lon_e)
        if e < w:
            e += 360.0
    
    # Ensure lat_s < lat_n
    if lat_s > lat_n:
        lat_s, lat_n = lat_n, lat_s
    
    return [w, e, lat_s, lat_n]

def convert_data_coordinates(lon_data, central_lon):
    """Convert data coordinates to match the projection central longitude."""
    if central_lon == 180.0:
        # Convert from 0-360 to -180-180 range
        lon_converted = np.where(lon_data > 180, lon_data - 360, lon_data)
    else:
        # Keep in 0-360 range
        lon_converted = lon_data
    
    return lon_converted

# -------------------- Auto-thinning --------------------
def span_from_extent(extent):
    """Compute longitudinal span from extent."""
    w, e, _, _ = extent
    lon_span = (e - w) if e >= w else (e + 360.0 - w)
    lat_span = abs(extent[3] - extent[2])
    return max(lon_span, lat_span)

def auto_plot_params(extent, nx, ny):
    """Determine plotting parameters based on extent and grid size."""
    span = span_from_extent(extent)

    if span >= 120:         # basin/hemisphere
        desired_x = 40
        barb_len  = 5
        mslp_lw   = 0.9
        coast_lw  = 0.9
        border_lw = 0.7
        state_lw  = 0.5
        cint      = 2
    elif span >= 60:        # sub-basin / multi-country
        desired_x = 55
        barb_len  = 6
        mslp_lw   = 1.0
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

# -------------------- Time helper --------------------
def read_valid_time(ds):
    """Extract valid time from dataset."""
    if hasattr(ds, 'variables'):
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
    else:
        return "Unknown valid time"

# -------------------- Main data retrieval --------------------
def retrieve_era5_data(year, month, day, hour, region_coords, api_key):
    """Retrieve ERA5 data for the specified region and time."""
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"
    
    # Get CDS areas for the region
    areas = region_to_cds_area(region_coords)
    
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)
    dataset = "reanalysis-era5-single-levels"
    
    base_request = {
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
    
    datasets = []
    temp_files = []
    
    try:
        # Retrieve data for each area
        for i, area in enumerate(areas):
            request = base_request.copy()
            request["area"] = area
            
            temp_file = tempfile.NamedTemporaryFile(suffix=f'_part{i}.nc', delete=False)
            temp_files.append(temp_file.name)
            temp_file.close()
            
            st.info(f"Retrieving data for area {i+1}/{len(areas)}: {area}")
            c.retrieve(dataset, request, temp_file.name)
            
            ds = nc.Dataset(temp_file.name)
            datasets.append(ds)
        
        # Merge datasets if multiple areas
        if len(datasets) == 1:
            ds = datasets[0]
            lon = ds.variables['longitude'][:]
            lat = ds.variables['latitude'][:]
            data = {
                'msl': ds.variables['msl'][:],
                'd2m': ds.variables['d2m'][:],
                'u10': ds.variables['u10'][:],
                'v10': ds.variables['v10'][:]
            }
        else:
            lon, lat, data = merge_datasets(datasets)
            ds = datasets[0]  # Use first dataset for metadata
        
        return ds, lon, lat, data
        
    finally:
        # Clean up datasets
        for ds in datasets:
            try:
                ds.close()
            except:
                pass
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

# -------------------- Main visualization --------------------
def generate_visualization(year, month, day, hour, region_coords, api_key):
    """Generate the ERA5 visualization."""
    
    # Retrieve data
    ds, lon_data, lat_data, data = retrieve_era5_data(year, month, day, hour, region_coords, api_key)
    
    try:
        # Extract variables and convert units
        mslp = data['msl'] / 100.0  # Convert to hPa
        d2m = data['d2m']           # Kelvin
        u10 = data['u10']
        v10 = data['v10']
        
        # Convert dewpoint to Fahrenheit
        dewpoint_f = (d2m - 273.15) * 9/5 + 32
        
        # Determine projection setup
        central_lon = get_projection_setup(region_coords)
        plot_extent = prepare_plot_extent(region_coords, central_lon)
        
        # Convert data coordinates to match projection
        lon_plot = convert_data_coordinates(lon_data, central_lon)
        
        # Sort data by longitude for consistent plotting
        sort_idx = np.argsort(lon_plot)
        lon_sorted = lon_plot[sort_idx]
        mslp_sorted = mslp[:, :, sort_idx]
        dewpoint_sorted = dewpoint_f[:, :, sort_idx]
        u10_sorted = u10[:, :, sort_idx]
        v10_sorted = v10[:, :, sort_idx]
        
        # Setup projection and figure
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        data_crs = ccrs.PlateCarree(central_longitude=central_lon)
        
        fig, ax = plt.subplots(figsize=(18, 9), subplot_kw={'projection': proj})
        ax.set_extent(plot_extent, crs=proj)
        
        # Auto-determine plotting parameters
        params = auto_plot_params(plot_extent, nx=len(lon_sorted), ny=len(lat_data))
        
        # Add map features
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=params['border_lw'])
        try:
            ax.add_feature(cfeature.STATES, linestyle=':', linewidth=params['state_lw'])
        except Exception:
            pass  # States layer may not be available
        
        # Plot dewpoint (filled contours)
        dp_slice = dewpoint_sorted[0, :, :]
        cf = ax.contourf(
            lon_sorted, lat_data, dp_slice,
            levels=np.linspace(-40, 90, 256), 
            cmap=create_custom_colormap(), 
            extend='both',
            transform=data_crs
        )
        
        # Plot pressure contours
        mslp_slice = mslp_sorted[0, :, :]
        cint = params['cint']
        mmin = np.floor(np.nanmin(mslp_slice) / cint) * cint
        mmax = np.ceil(np.nanmax(mslp_slice) / cint) * cint
        levels = np.arange(mmin, mmax + cint, cint)
        
        # Limit number of contour levels
        if len(levels) > 60:
            skip = int(np.ceil(len(levels) / 60))
            levels = levels[::skip]
        
        ax.contour(
            lon_sorted, lat_data, mslp_slice,
            levels=levels, colors='black',
            linewidths=params['mslp_lw'],
            transform=data_crs
        )
        
        # Plot wind barbs
        LON2, LAT2 = np.meshgrid(lon_sorted, lat_data)
        si = params['stride_y']
        sj = params['stride_x']
        
        ax.barbs(
            LON2[::si, ::sj], LAT2[::si, ::sj],
            u10_sorted[0, ::si, ::sj], v10_sorted[0, ::si, ::sj],
            length=params['barb_len'],
            transform=data_crs
        )
        
        # Add colorbar and title
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.7)
        cb.set_label('2m Dewpoint Temperature (°F)')
        
        date_str = read_valid_time(ds)
        ax.set_title(f'ERA5 Pressure, Dewpoint, and Wind\nValid for: {date_str}\nPlotted by Sekai Chandra (@Sekai_WX)')
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        
        plt.close(fig)
        return buffer
        
    finally:
        try:
            ds.close()
        except:
            pass

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

# Region selection
col5, col6 = st.columns(2)
with col5:
    region_options = []
    for category, regions in REGIONS.items():
        for region_name in regions.keys():
            region_options.append(f"{category}: {region_name}")
    selected_region = st.selectbox("Select Region", region_options)

with col6:
    generate_button = st.button("Generate Visualization", type="primary")

# Generate visualization
if generate_button:
    category, region_name = selected_region.split(": ", 1)
    region_coords = REGIONS[category][region_name]
    
    # Show region info
    lon_w, lon_e, lat_s, lat_n = region_coords
    crosses_idl = crosses_dateline(lon_w, lon_e)
    
    if crosses_idl:
        st.info(f"🌏 Selected region crosses the International Date Line. Using optimized data retrieval.")
    
    try:
        with st.spinner("Downloading ERA5 data and generating visualization..."):
            image_buffer = generate_visualization(year, month, day, hour, region_coords, api_key)
        
        st.success("✅ Visualization generated successfully!")
        st.image(image_buffer, caption=f"ERA5 Weather Data for {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC - {region_name}")
        
        # Download button
        st.download_button(
            label="📥 Download Image",
            data=image_buffer,
            file_name=f"ERA5_{year}{month:02d}{day:02d}{hour:02d}_{region_name.replace(' ', '_')}.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"❌ Error generating visualization: {str(e)}")
        st.info("Please check that the date/time is valid and the CDS API service is available.")
        
        # Debug info
        with st.expander("Debug Information"):
            st.write(f"Region coordinates: {region_coords}")
            st.write(f"Crosses dateline: {crosses_idl}")
            st.write(f"CDS areas: {region_to_cds_area(region_coords)}")