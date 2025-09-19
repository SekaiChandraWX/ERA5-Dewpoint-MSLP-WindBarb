# ERA5 Weather Visualization

A Streamlit web application for generating weather visualizations using ERA5 reanalysis data from the Copernicus Climate Data Store (CDS).

## Features

- Generate weather maps showing pressure, dewpoint temperature, and wind data
- Support for multiple global regions including Continental US, Europe, Asia, and tropical basins
- Interactive date/time selection (hourly data from 1940 to present)
- Custom dewpoint temperature colormap
- Downloadable high-resolution images

## Prerequisites

### CDS API Key
You'll need a free account and API key from the Copernicus Climate Data Store:

1. Register at [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
2. Go to [https://cds.climate.copernicus.eu/api-how-to](https://cds.climate.copernicus.eu/api-how-to)
3. Copy your API key from the webpage

## Local Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd era5-weather-visualization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
3. Click "New app"
4. Connect your GitHub repository
5. Set the main file path to `app.py`
6. Deploy

## Usage

1. Enter your CDS API key in the text field
2. Select the year, month, day, and hour for the data you want to visualize
3. Choose a region from the dropdown menu:
   - **General regions**: Continental US, Europe, Asia, etc.
   - **Tropical basins**: Various ocean basins for tropical cyclone monitoring
4. Click the red "Generate" button
5. Wait for the data to download and the visualization to generate
6. Download the resulting image if desired

## Supported Regions

### General
- Continental United States
- North Atlantic Basin
- Europe
- Middle East and South Asia
- East and Southeast Asia
- Australia and Oceania

### Tropics
- West Pacific Basin
- East Pacific Basin
- Central Pacific Basin
- Northern Indian Ocean Basin
- South Indian Ocean Basin
- Australian Basin

## Data Information

- **Source**: ERA5 Reanalysis from ECMWF/Copernicus Climate Data Store
- **Variables**: Mean sea level pressure, 2m dewpoint temperature, 10m wind components
- **Resolution**: Hourly data on a regular lat/lon grid
- **Coverage**: Global, from 1940 to near real-time

## Technical Details

The application downloads NetCDF data from the CDS API, processes it using Python scientific libraries, and generates publication-quality weather maps with:
- Pressure contours (isobars) in 2 hPa intervals
- Dewpoint temperature filled contours with custom colormap
- Wind barbs showing direction and speed
- Geographic boundaries and coastlines

## Credits

Original visualization code by Sekai Chandra (@Sekai_WX)
Streamlit adaptation for web deployment

## License

This project is open source and available under the MIT License.