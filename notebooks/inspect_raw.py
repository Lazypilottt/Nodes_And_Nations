# After downloading manually, inspect the file:
import pandas as pd

# The Excel has one sheet per year (1990, 1995 ... 2020)
xl = pd.ExcelFile("/Users/lazypilot/Desktop/Lazypilot/IIT Bhilai/Acad/SEM6/NS/Nodes and Nations/data/raw/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
print("Sheet names:", xl.sheet_names)  # should see years as sheet names

# Read one table to understand the structure
df_sample = pd.read_excel(xl, sheet_name="Table 1", header=10, index_col=0)
print("Shape:", df_sample.shape)   # expect ~232 x 232
print("Columns:", df_sample.columns.tolist()[:10])  # First 10 columns
print("Head:")
print(df_sample.head(10))

# Check for network capability: look for country-to-country pairs
print("\nChecking for bilateral migration data capability:")
print("Unique destinations:", df_sample['Region, development group, country or area of destination'].nunique())
print("Unique origins:", df_sample['Region, development group, country or area of origin'].nunique())

# Filter for actual country pairs (not World or regions)
country_pairs = df_sample[
    (df_sample['Region, development group, country or area of destination'] != 'World') &
    (df_sample['Region, development group, country or area of origin'] != 'World') &
    (~df_sample['Region, development group, country or area of destination'].str.contains('regions|group', case=False, na=False)) &
    (~df_sample['Region, development group, country or area of origin'].str.contains('regions|group', case=False, na=False))
]
print("Country-to-country pairs:", len(country_pairs))
print("Sample country pairs:")
print(country_pairs[['Region, development group, country or area of destination', 'Region, development group, country or area of origin', 2024]].head(10))

# Check data types and missing values
print("\nData quality check:")
print("Data types:")
print(df_sample.dtypes)
print("Missing values per column:")
print(df_sample.isnull().sum())