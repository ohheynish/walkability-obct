
This repository contains the scripts used in the analysis for our paper: [A Walk across Europe: Development of a high-resolution walkability index](https://arxiv.org/abs/2504.17897).

## Code (Jupyter Notebook) Execution Order and Description

The recommended order of execution with brief descriptions:

1. **01_generate_grids.ipynb**  
   Generates 100m x 100m grids from the Eurostat 100km x 100km grid file.

2. **02_*** notebooks  
   Collect raw data from various sources and process walkability components for 100m x 100m grids:
   - **02_sidewalks.ipynb**: using [`OSMnx`](https://github.com/gboeing/osmnx).
   - **02_sidewalks_osmium.ipynb**: Efficient sidewalk processing with [`osmium`](https://osmcode.org/pyosmium/).
   - **02_street_intersections.ipynb**: using `OSMnx`.
   - **02_street_intersections_osmium.ipynb**: Efficient intersection calculation with `osmium`.
   - **02_green_spaces.ipynb**: Extracts NDVI from Sentinel-2 (requires Google Earth Engine & Google Cloud).
   - **02_slope.ipynb**: Extracts slope from NASA DEM (requires Google Earth Engine & Google Cloud).
   - **02_public_transport.ipynb**: Counts OSM public transport features (requires Overpass API).
   - **02_public_transport_osmium.ipynb**: Efficient version using `osmium`.
   - **02_landuse_mix.ipynb**: Computes land use mix (entropy) using Copernicus LULC data (requires Google Earth Engine & Google Cloud).
   - **02_degree_of_urbanization.ipynb**: Uses GHSL data to assign urbanization class (requires Google Earth Engine & Google Cloud).
   - **02_population.ipynb**: Assigns population counts from GHSL (requires Google Earth Engine & Google Cloud).

3. **03_generate_isochrones.ipynb**  
   Uses [`Valhalla`](https://github.com/valhalla/valhalla) routing engine to generate isochrone polygons (also a component: 15-min walking area).

4. **04_isochrones_to_gpu.ipynb**  
   Converts isochrones to array format for GPU-friendly processing.

5. **05_process_isochrones_gpu.ipynb**  
   Runs the core GPU kernel to compute distance-decayed metrics inside isochrones (requires NVIDIA GPU).

6. **06_walk_index.ipynb**  
   Computes the final walkability index using all the processed components.

## New vs. Old Scripts

Scripts with **`osmium`** in the filename are newly developed for efficient large-scale processing of OSM data. Earlier versions relied on `OSMnx` and Overpass API, which are inefficient for bulk processing.

## Dependencies (in addition to libraries)

- **Google Earth Engine**: Requires an [Earth Engine account](https://signup.earthengine.google.com/) and setup of the Python API.
- **Google Cloud Storage**: Temporary storage for Earth Engine exports. Requires a Google Cloud account and service account key.
- **Valhalla**: Isochrone generation via open-source `Valhalla` routing engine. 
- **NVIDIA GPU**: Required for processing isochrones on GPU (step 05).

## Required Downloads

Some scripts expect the following data to be pre-downloaded:

- `europe-latest.osm.pbf` – Raw OpenStreetMap data for Europe.  
  Download from: [Geofabrik](https://download.geofabrik.de/europe.html)

## Interactive Web-Atlas (coming soon!)

## Environment Setup

To create and activate the environment for reproducibility:

```bash
conda env create -f environment.yml
conda activate geo_env
```

## Additional Info

If you'd like to use the already processed data for your projects, don't hesitate to reach out!  
This work was done as part of the [`OBCT`](https://www.obct.nl/) project.
