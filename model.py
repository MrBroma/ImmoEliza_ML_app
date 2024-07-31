import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

def load_data(columns_to_keep):
    try:
        # Load the dataset
        data = pd.read_csv("data/dataset_sales_cleaned.csv")
        data_sales = data[columns_to_keep]
        return data_sales
    except FileNotFoundError:
        print("The specified file was not found.")
        return None

def prepare_data(data_sales):
    # List for ordinal encoding
    ode_cols = ["State_Encoded", "PEB_Encoded"]
    
    # List for one-hot encoding
    ohe_cols = ["District", "Locality", "Province", "Region",
                "SubtypeOfProperty", "Kitchen", "Garden",
                "SwimmingPool", "Terrace"]
    
    # List of numerical columns
    num_cols = ["BathroomCount", "BedroomCount", "ConstructionYear", "GardenArea",
                "LivingArea", "NumberOfFacades", "RoomCount", "ShowerCount", "SurfaceOfPlot", "FloodingZone_Encoded"]
    
    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ('impute', KNNImputer()),
        ('scaler', StandardScaler())
    ])
    
    # Ordinal encoding pipeline
    ode_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler()) 
    ])
    
    # One-hot encoding pipeline
    ohe_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer for different feature types
    col_trans = ColumnTransformer(transformers=[
        ('num_p', num_pipeline, num_cols),
        ('ode_p', ode_pipeline, ode_cols),
        ('ohe_p', ohe_pipeline, ohe_cols)
    ], remainder='passthrough', n_jobs=-1)
    
    return col_trans

def train_model(data_sales, col_trans):
    # Separate features and target variable
    X = data_sales.drop('Price', axis=1)
    y = data_sales['Price']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Define the pipeline with preprocessing and the XGBoost model
    pipeline = Pipeline(steps=[
        ('preprocessing', col_trans),
        ('model', XGBRegressor(tree_method='gpu_hist', use_label_encoder=False, eval_metric='rmse'))  # Use GPU
    ])

    # Define parameter grid for randomized search
    param_distributions = {
        'preprocessing__num_p__impute__n_neighbors': [3, 5],  # Reduce grid size for quicker testing
        'model__n_estimators': [50, 100],  # Reduce number of trees
        'model__max_depth': [3, 5],  # Depth of the trees
        'model__learning_rate': [0.1, 0.05],  # Learning rate
    }

    # Use RandomizedSearchCV to find the best parameters
    randomized_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=10,  # Number of parameter combinations to try
        scoring='neg_mean_absolute_error',  # Use MAE as the scoring method
        cv=3,  # Use 3-fold cross-validation
        verbose=1,  # Increase verbosity to see the process
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit randomized search
    randomized_search.fit(X_train, y_train)

    # Retrieve the best parameters and set them in the pipeline
    best_params = randomized_search.best_params_
    print("Best parameters found: ", best_params)

    pipeline.set_params(
        preprocessing__num_p__impute__n_neighbors=best_params['preprocessing__num_p__impute__n_neighbors'],
        model__n_estimators=best_params['model__n_estimators'],
        model__max_depth=best_params['model__max_depth'],
        model__learning_rate=best_params['model__learning_rate'],
    )

    # Fit the final pipeline with the best parameters
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    return train_score, test_score, mae

# Variables
columns_to_keep = [
    "BathroomCount",
    "BedroomCount",
    "ConstructionYear",
    "District",
    "Garden",
    "GardenArea",
    "Kitchen",
    "LivingArea",
    "Locality",
    "NumberOfFacades",
    "Price",
    "Province",
    "Region",
    "RoomCount",
    "ShowerCount",
    "SubtypeOfProperty",
    "SurfaceOfPlot",
    "SwimmingPool",
    "Terrace",
    "PEB_Encoded",
    "State_Encoded",
    "FloodingZone_Encoded"
]

# Load and prepare data
df_sales = load_data(columns_to_keep)

if df_sales is not None:
    data_preparation = prepare_data(df_sales)

    # Train model and evaluate performance
    train_score, test_score, mae = train_model(df_sales, data_preparation)

    print("Train Score: ", train_score)
    print("Test Score: ", test_score)
    print("Mean Absolute Error: ", mae)
