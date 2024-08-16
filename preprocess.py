import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
df = pd.read_csv("data/dataset_sales_cleaned.csv")

# Define target and feature columns
target = ["Price"]
int_cols = ["BedroomCount", "BathroomCount", "ConstructionYear", "NumberOfFacades", "RoomCount", "ShowerCount"]
float_cols = ["LivingArea", "GardenArea", "SurfaceOfPlot"]
categorical_cols = ["District", "Kitchen", "Locality", "Province", "Region", "SubtypeOfProperty", "SwimmingPool", "Terrace", "PEB", "StateOfBuilding", "FloodingZone", "Furnished"]

# Combine all feature columns
features_to_keep = int_cols + float_cols + categorical_cols
print(f"Number of columns to keep: {len(features_to_keep)}\n")

# Define the preprocessing for different types of columns
preprocessor = ColumnTransformer(
    transformers=[
        ("int", SimpleImputer(strategy="most_frequent"), int_cols),
        ("float", SimpleImputer(strategy="mean"), float_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder="passthrough"
)

# Apply the preprocessing
X = preprocessor.fit_transform(df[features_to_keep])

# Extract feature names
feature_names = preprocessor.get_feature_names_out()

# Create a DataFrame with the transformed data
df_preprocessed = pd.DataFrame(X, columns=feature_names)
print(df_preprocessed.head())

# Save the preprocessor
joblib.dump(preprocessor, "models/preprocessor_compressed.joblib", compress=3)
print("Preprocessor Saved!")

