# Climate Data Analysis Project

## 📌 Project Overview

This project performs a comprehensive climate data analysis using **point-based station data** and **gridded CPC datasets** (2000–2024). The analysis focuses on four diverse locations:

* **Corpus Christi, TX, USA**
* **New Delhi, India**
* **San Francisco, CA, USA**
* **Chicago, IL, USA**

The workflow includes data extraction, preprocessing, merging, and a suite of exploratory and statistical analyses, producing both **time-series CSV files** and **visualization outputs**.


## 📂 Directory Structure

```
DA Assignment/
│
├── data extraction/              # Raw data extraction and preprocessing
│   ├── outputs/                  # Extracted CSVs and NetCDF files
│   │   ├── *_temp_monthly.csv
│   │   ├── *_precip_monthly.csv
│   │   ├── *_slp_monthly.csv
│   │   ├── *_rhum_monthly.csv
│   │   ├── *_all_variables.csv
│   │   ├── precip_grid_2000_2024.nc
│   │   └── temp_grid_2000_2024.nc
│   ├── point_extract.py
│   ├── gridded_extraction_only.py
│   └── additional_pt.py
│
├── preprocessing/
│   └── analysis.py               # Main analysis pipeline
│
├── results/                      # Final outputs and figures
│   ├── assign1-locations.png
│   ├── regional_grid_maps.png
│   ├── point_vs_grid_comparison.png
│   ├── seasonal_analysis.png
│   ├── periodic_analysis.png
│   ├── correlation_analysis.png
│   ├── distribution_analysis.png
│   ├── additional_variables_timeseries.png
│   ├── summary_statistics.csv
│   └── *_all_variables.csv
```


## ⚙️ Workflow

### **1. Data Extraction**

* **Point-based data** extracted for each city (`point_extract.py`)

  * Temperature (°C)
  * Precipitation (mm/month)
* **Gridded data** (`gridded_extraction_only.py`)

  * Global CPC temperature & precipitation fields (2000–2024)
  * Saved as NetCDF (`temp_grid_2000_2024.nc`, `precip_grid_2000_2024.nc`).
* **Additional Variables** (`additional_pt.py`)

  * Extracts **Sea Level Pressure (hPa)** and **Relative Humidity (%)** for all four cities.
  * Ensures compliance with assignment requirement to include **two extra climate variables**.


### **2. Preprocessing**

* `analysis.py` merges city-level variables (Temp, Precip, SLP, RHUM) into unified CSV files (`*_all_variables.csv`).
* Ensures consistent formatting for downstream analysis.

### **3. Analysis Modules**

The pipeline (`analysis.py`) generates the following outputs:

* **Time Series Plots**

  * `assign1-locations.png` → Temperature & precipitation trends (2000–2024).

* **Regional Grid Maps**

  * `regional_grid_maps.png` → 16×16 regional grids centered on each city.

* **Point vs Grid Comparison**

  * `point_vs_grid_comparison.png` → Validates consistency between extracted point and gridded data.

* **Seasonal Analysis**

  * `seasonal_analysis.png` → Monthly mean cycles (temperature vs precipitation).

* **Periodic Analysis**

  * `periodic_analysis.png` → Frequency spectrum using FFT, highlighting annual cycles.

* **Correlation Analysis**

  * `correlation_analysis.png` → Heatmaps showing inter-variable relationships per city.

* **Distribution Analysis**

  * `distribution_analysis.png` → Histograms + fitted normal curves for each variable.

* **Additional Variables**

  * `additional_variables_timeseries.png` → Sea-level pressure & relative humidity trends.

* **Summary Statistics**

  * `summary_statistics.csv` → Mean, standard deviation, min, max for each variable.


## 📊 Results Highlights

* **Seasonality**:

  * New Delhi shows strong monsoonal precipitation peaks (June–August).
  * Chicago has strong temperature seasonality with winter lows below 0°C.

* **Correlations**:

  * Negative correlation between **temperature** and **sea-level pressure** is consistent across sites.
  * Positive relationship between **precipitation** and **humidity** is evident.

* **Periodic Analysis**:

  * All cities exhibit a strong **annual cycle** (frequency ≈ 1 cycle/year).

* **Distribution Analysis**:

  * Temperature distributions are near-normal, while precipitation is highly skewed.

* **Validation**:

  * Point vs grid comparison shows near-perfect agreement, confirming extraction accuracy.


## 🚀 How to Run

1. Navigate to `preprocessing/`
2. Run the main analysis script:

```bash
python3 analysis.py
```

3. All results will be saved in the `results/` directory.


## ✅ Deliverables

* **REQUIRED Outputs:**

  * `assign1-locations.png` (time series)
  * `regional_grid_maps.png` (gridded maps)
  * `point_vs_grid_comparison.png` (validation)

* **ADDITIONAL Outputs:**

  * Seasonal, periodic, correlation, and distribution analyses
  * Additional variable timeseries
  * Summary statistics table


## 📌 Notes

* The project uses **xarray**, **pandas**, **matplotlib**, **seaborn**, and **scipy**.
* Latitude/longitude handling accounts for both `-180–180` and `0–360` systems.
* Regional maps use \~16×16 grids (\~8°×8° window).
