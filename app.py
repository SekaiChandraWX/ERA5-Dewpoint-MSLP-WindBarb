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

# Region coordinates dictionary
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
        "West Pacific Basin": [94.9, 183.5, -14.6, 56.1],
        "East Pacific Basin": [-161.4, -86.3, 3, 39],
        "Central Pacific Basin": [-188.8, -141.6, 2.4, 41.1],
        "Northern Indian Ocean Basin": [-317, -256.3, -5, 34],
        "South Indian Ocean Basin": [32.7, 125.4, -44.8, 3.5],
        "Australian Basin": [100, 192.7, -50.2, -1.9]
    }
}

def create_custom_colormap():
    """Create the custom colormap for dewpoint temperature."""
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
    
    # Normalize the colors
    norm_colors = [(r/255, g/255, b/255) for r, g, b in colors]
    return mcolors.LinearSegmentedColormap.from_list('custom_dewpoint', norm_colors, N=256)

def generate_visualization(year, month, day, hour, region_coords, api_key):
    """Generate the ERA5 visualization."""
    
    date_input = f"{year:04}{month:02}{day:02}{hour:02}"
    
    # Configure CDS API
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=api_key)
    
    # Define dataset and request parameters
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
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        target = tmp_file.name
    
    try:
        # Retrieve data from ECMWF
        c.retrieve(dataset, request, target)
        
        # Load the data
        dataset_nc = nc.Dataset(target)
        
        # Extract the variables
        mslp = dataset_nc.variables['msl'][:] / 100.0  # Convert to hPa
        dewpoint = (dataset_nc.variables['d2m'][:] - 273.15) * 9/5 + 32  # Convert to Fahrenheit
        temperature = (dataset_nc.variables['t2m'][:] - 273.15) * 9/5 + 32  # Convert to Fahrenheit
        u_wind = dataset_nc.variables['u10'][:]
        v_wind = dataset_nc.variables['v10'][:]
        lon = dataset_nc.variables['longitude'][:]
        lat = dataset_nc.variables['latitude'][:]
        
        # Extract the datetime variable and format it
        valid_time = dataset_nc.variables['valid_time']
        time_unit = valid_time.units
        time_calendar = valid_time.calendar
        time_val = valid_time[:]
        date = nc.num2date(time_val[0], units=time_unit, calendar=time_calendar)
        date_str = date.strftime("%B %d, %Y - %H:%M UTC")
        
        # Create custom colormap
        cmap = create_custom_colormap()
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(18, 9), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(region_coords, crs=ccrs.PlateCarree())
        ax.set_facecolor('#C0C0C0')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':')
        
        # Plot MSLP isobars
        cs = ax.contour(lon, lat, mslp[0, :, :], levels=np.arange(950, 1050, 2), colors='black', linewidths=1)
        
        # Plot 2m dewpoint temperature with custom colormap
        levels = np.linspace(-40, 90, 256)  # Use 256 levels for smooth colormap
        cf = ax.contourf(lon, lat, dewpoint[0, :, :], cmap=cmap, levels=levels, extend='both')
        
        # Plot 10m wind barbs
        ax.barbs(lon[::5], lat[::5], u_wind[0, ::5, ::5], v_wind[0, ::5, ::5], length=6)
        
        # Add colorbar for dewpoint
        cb = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.7)
        cb.set_label('2m Dewpoint Temperature (°F)')
        
        # Add title
        ax.set_title(f'ERA5 Pressure, Dewpoint, and Wind\nValid for: {date_str}\nPlotted by Sekai Chandra (@Sekai_WX)')
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        
        # Clean up
        dataset_nc.close()
        plt.close(fig)
        
        return buffer
        
    finally:
        # Clean up temporary file
        if os.path.exists(target):
            os.remove(target)

# Streamlit UI
st.title("ERA5 Weather Visualization")

# Get API key from secrets
try:
    api_key = st.secrets["CDS_API_KEY"]
except KeyError:
    st.error("CDS API key not found in secrets. Please configure your API key.")
    st.stop()

# Date and time inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    year = st.number_input("Year", min_value=1940, max_value=datetime.now().year, value=2023)

with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=1)

with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

with col4:
    hour = st.number_input("Hour", min_value=0, max_value=23, value=12)

# Region selection and generate button
col5, col6 = st.columns(2)

with col5:
    # Create region options for selectbox
    region_options = []
    for category, regions in REGIONS.items():
        for region_name in regions.keys():
            region_options.append(f"{category}: {region_name}")
    
    selected_region = st.selectbox("Select Region", region_options)

with col6:
    generate_button = st.button("Generate", type="primary", 
                               help="Generate the ERA5 visualization")

# Generate visualization when button is clicked
if generate_button:
    # Parse selected region
    category, region_name = selected_region.split(": ", 1)
    region_coords = REGIONS[category][region_name]
    
    try:
        with st.spinner("Downloading ERA5 data and generating visualization..."):
            image_buffer = generate_visualization(year, month, day, hour, region_coords, api_key)
            
        st.success("Visualization generated successfully!")
        st.image(image_buffer, caption=f"ERA5 Weather Data for {year}-{month:02d}-{day:02d} {hour:02d}:00 UTC")
        
        # Download button
        st.download_button(
            label="Download Image",
            data=image_buffer,
            file_name=f"ERA5_{year}{month:02d}{day:02d}{hour:02d}_{region_name.replace(' ', '_')}.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.info("Make sure the date/time is valid and the API service is available.")