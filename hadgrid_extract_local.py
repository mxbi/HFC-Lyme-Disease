import xarray as xr
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Configuration
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  # All available years
VARS = ["hurs", "tasmax", "sun", "tasmin", "tas", "snowLying", "groundfrost"]  # Both available variables

# Local paths
DATA_DIR = Path("hadgrid-local")
LA_SHP = Path("Local_Authority_Districts_December_2024_Boundaries_UK_BGC_9075602114651001607/LAD_DEC_24_UK_BGC.shp")
OUT = "lad_climate_data_2017_2024.csv"

def get_local_file_path(variable, year):
    """Get the local file path for a given variable and year."""
    filename = f"{variable}_hadukgrid_uk_12km_ann_{year}01-{year}12.nc"
    return DATA_DIR / filename

# ------------------------------------------------------------------
# 1. Load Local Authority Districts shapefile
# ------------------------------------------------------------------
print("📍 Loading Local Authority Districts...")
lads = gpd.read_file(LA_SHP)
if lads.crs is None:
    lads = lads.set_crs(27700)
lads = lads.to_crs(27700)  # Ensure we're in British National Grid

# Get representative points (centroids) for sampling
lads["geometry"] = lads.geometry.representative_point()
xi = lads.geometry.x.values
yi = lads.geometry.y.values

print(f"✓ Loaded {len(lads)} Local Authority Districts")
print(f"📋 Sample councils: {', '.join(lads['LAD24NM'].head(5).tolist())}")

# ------------------------------------------------------------------
# 2. Process climate data from local files
# ------------------------------------------------------------------
all_data = []

for year in YEARS:
    print(f"\n📅 Processing year {year}...")
    year_data = {"year": year}
    
    for var in VARS:
        local_path = get_local_file_path(var, year)
        
        # Check if file exists
        if not local_path.exists():
            print(f"⚠️  File not found: {local_path}")
            continue
        
        try:
            print(f"📖 Reading {local_path.name}...")
            
            # Open the dataset
            ds = xr.open_dataset(local_path, engine='h5netcdf')
            da = ds[var]
            
            # Detect spatial dimension names
            spatial_dims = [d for d in da.dims if d not in ("time",)]
            if "projection_x_coordinate" in spatial_dims and "projection_y_coordinate" in spatial_dims:
                xdim, ydim = "projection_x_coordinate", "projection_y_coordinate"
            else:
                # Fallback to first two non-time dimensions
                xdim, ydim = spatial_dims[0:2]
            
            # Sample at Local Authority centroids
            # For annual data, there should be one time step per year
            if "time" in da.dims and len(da.time) > 0:
                # Take the annual value (should be only one per file)
                da_annual = da.isel(time=0) if len(da.time) == 1 else da.mean(dim="time")
            else:
                da_annual = da
            
            vals = da_annual.sel({
                xdim: xr.DataArray(xi, dims="pts"),
                ydim: xr.DataArray(yi, dims="pts")
            }, method="nearest").values
            
            year_data[var] = vals
            print(f"✓ Extracted {var} for {year}")
            
            ds.close()
            
        except Exception as e:
            print(f"❌ Error processing {var} for {year}: {e}")
            continue
    
    # If we have any variable data for this year, add it to results
    if len(year_data) > 1:  # More than just the year
        all_data.append(year_data)

# ------------------------------------------------------------------
# 3. Create final DataFrame
# ------------------------------------------------------------------
print("\n📊 Creating final dataset...")

results = []
for year_data in all_data:
    year = year_data["year"]
    
    for i, (_, lad_row) in enumerate(lads.iterrows()):
        row = {
            "year": year,
            "council_name": lad_row["LAD24NM"]
        }
        
        # Add climate variables
        for var in VARS:
            if var in year_data:
                row[var] = year_data[var][i]
            else:
                row[var] = None
        
        results.append(row)

# Convert to DataFrame and save
df = pd.DataFrame(results)

if len(df) > 0:
    df.to_csv(OUT, index=False)
    print(f"\n✅ Successfully created {OUT}")
    print(f"📈 Dataset contains {len(df)} rows ({len(df['year'].unique())} years × {len(df['council_name'].unique())} councils)")
    print(f"🏛️  Sample councils: {', '.join(df['council_name'].unique()[:5])}")
    print(f"📅 Years: {sorted(df['year'].unique())}")
    print(f"🌡️  Variables: {[col for col in df.columns if col not in ['year', 'council_name']]}")
    
    # Show a few example rows
    print(f"\n📝 Sample data:")
    print(df.head(10).to_string(index=False))
    
    # ------------------------------------------------------------------
    # 4. Compare council names with forecast.csv
    # ------------------------------------------------------------------
    print(f"\n🔍 Comparing council names with forecast.csv...")
    
    try:
        forecast_path = Path("submission/forecast.csv")
        if forecast_path.exists():
            forecast_df = pd.read_csv(forecast_path)
            
            # Get unique council names from both datasets
            climate_councils = set(df['council_name'].unique())
            forecast_councils = set(forecast_df['Council'].unique())
            
            print(f"📊 Council name comparison:")
            print(f"   Climate data: {len(climate_councils)} unique councils")
            print(f"   Forecast data: {len(forecast_councils)} unique councils")
            
            # Define manual mapping for known fuzzy matches
            council_mapping = {
                "Bournemouth, Christchurch and Poole": "Bournemouth",
                "Bristol, City of": "Bristol",
                "Buckinghamshire": "Buckinghamshire UA",
                "Folkestone and Hythe": "Folkestone & Hythe",
                "Herefordshire, County of": "Herefordshire", 
                "Kingston upon Hull, City of": "Kingston upon Hull",
                "North Yorkshire": "North Yorkshire UA",
                "Somerset": "Somerset UA"
            }
            
            # Find councils in climate data but not in forecast
            climate_only = climate_councils - forecast_councils
            if climate_only:
                print(f"\n❌ Councils in climate data but NOT in forecast.csv ({len(climate_only)}):")
                for council in sorted(climate_only):
                    if council in council_mapping:
                        print(f"   - {council} → will map to '{council_mapping[council]}'")
                    else:
                        print(f"   - {council}")
            else:
                print(f"\n✅ All climate councils found in forecast.csv")
            
            # Find councils in forecast but not in climate data
            forecast_only = forecast_councils - climate_councils
            if forecast_only:
                print(f"\n❌ Councils in forecast.csv but NOT in climate data ({len(forecast_only)}):")
                for council in sorted(forecast_only):
                    # Check if this council has a mapped source
                    source_council = next((k for k, v in council_mapping.items() if v == council), None)
                    if source_council:
                        print(f"   - {council} ← will get data from '{source_council}'")
                    else:
                        print(f"   - {council}")
            else:
                print(f"\n✅ All forecast councils found in climate data")
            
            # Show exact matches
            exact_matches = climate_councils & forecast_councils
            print(f"\n✅ Exact matches: {len(exact_matches)} councils")
            
            # Add duplicate rows for mapped councils
            print(f"\n🔄 Adding duplicate rows for mapped councils...")
            additional_rows = []
            
            for source_council, target_council in council_mapping.items():
                if source_council in climate_councils and target_council in forecast_only:
                    # Get all rows for the source council
                    source_rows = df[df['council_name'] == source_council].copy()
                    # Update the council name to match forecast.csv
                    source_rows['council_name'] = target_council
                    additional_rows.append(source_rows)
                    print(f"   ✓ Copied {len(source_rows)} rows: '{source_council}' → '{target_council}'")
            
            if additional_rows:
                # Combine additional rows and add to main dataframe
                additional_df = pd.concat(additional_rows, ignore_index=True)
                df_extended = pd.concat([df, additional_df], ignore_index=True)
                
                # Save the extended dataset
                extended_filename = OUT.replace('.csv', '_extended.csv')
                df_extended.to_csv(extended_filename, index=False)
                
                print(f"\n✅ Created extended dataset: {extended_filename}")
                print(f"📈 Extended dataset contains {len(df_extended)} rows ({len(df_extended['council_name'].unique())} unique councils)")
                
                # Final comparison
                extended_climate_councils = set(df_extended['council_name'].unique())
                still_missing = forecast_councils - extended_climate_councils
                if still_missing:
                    print(f"\n⚠️  Still missing from climate data ({len(still_missing)}):")
                    for council in sorted(still_missing):
                        print(f"   - {council}")
                else:
                    print(f"\n🎉 All forecast councils now have climate data!")
                
            else:
                print(f"   No mappings to apply.")
                
        else:
            print(f"⚠️  forecast.csv not found at {forecast_path}")
            
    except Exception as e:
        print(f"❌ Error comparing with forecast.csv: {e}")

else:
    print(f"\n❌ No data was successfully processed. Check the error messages above.")