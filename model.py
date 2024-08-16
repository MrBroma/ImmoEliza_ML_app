from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import multiprocessing

def print_score(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    r_2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(f"""
    ################# Score for {model_name} ###############
        R²  =   {r_2:.4f}
        MAE =   {mae:.4f}
        
    #######################################################    
""")

# Load the preprocessor and data
preprocessor = joblib.load("models/preprocessor_compressed.joblib")
df = pd.read_csv("data/dataset_sales_cleaned.csv")

# Separate features and target
X = df[preprocessor.feature_names_in_]
y = df["Price"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Transform the data using the preprocessor
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

# Test Random Forest Regressor
print("Testing Random Forest Regressor")
rf_model = RandomForestRegressor(n_jobs=num_cores)  # Use all available cores
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print_score(y_test, y_pred_rf, "Random Forest")
joblib.dump(rf_model, "models/random_forest_compressed.joblib", compress=3)

# Test XGBoost Regressor
print("Testing XGBoost Regressor")
xgb_model = xgb.XGBRegressor(
    n_jobs=num_cores,  # Use all available cores
    objective='reg:squarederror',  # Use squared error for regression
    verbosity=1,  # Output progress
    random_state=42,  # Ensure reproducibility
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print_score(y_test, y_pred_xgb, "XGBoost")
joblib.dump(xgb_model, "models/xgboost_compressed.joblib", compress=3)
