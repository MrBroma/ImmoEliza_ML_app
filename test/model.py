import os
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

def load_data(file_path, columns_to_keep):
    if not os.path.exists(file_path):
        print("File not found")
        return None

    data = pd.read_csv(file_path)
    return data[columns_to_keep]

def preprocess_data_with_dummies(data_sales):
    # Extract categorical columns
    categorical_cols = ["District", "Locality", "Province", "Region",
                        "SubtypeOfProperty", "Kitchen", "Garden",
                        "SwimmingPool", "Terrace", "Furnished"]

    # Create dummy variables for categorical features
    data_sales_dummies = pd.get_dummies(data_sales, columns=categorical_cols, drop_first=True)

    return data_sales_dummies

def save_preprocessor_and_unique_values(data_sales):
    os.makedirs('models', exist_ok=True)
    
    # Extract categorical columns
    categorical_cols = ["District", "Locality", "Province", "Region",
                        "SubtypeOfProperty", "Kitchen", "Garden",
                        "SwimmingPool", "Terrace", "Furnished"]

    # Save unique categorical values
    unique_values = {col: data_sales[col].unique().tolist() for col in categorical_cols}
    
    with open('models/unique_values.json', 'w') as file:
        json.dump(unique_values, file)
    print("Unique values saved in 'models/unique_values.json'")

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test):
    # Define the XGBoost model
    xgb_model = XGBRegressor(tree_method='hist', use_label_encoder=False, eval_metric='rmse')
    
    # Define hyperparameter search space
    param_distributions_xgb = {
        'model__n_estimators': randint(50, 200),
        'model__max_depth': randint(3, 10),
        'model__learning_rate': uniform(0.01, 0.3),
    }
    
    # Create a pipeline with the model
    pipeline_xgb = Pipeline(steps=[
        ('model', xgb_model)
    ])
    
    # Perform randomized search
    randomized_search_xgb = RandomizedSearchCV(
        pipeline_xgb,
        param_distributions_xgb,
        n_iter=10,
        scoring='neg_mean_absolute_error',
        cv=2,
        verbose=1,
        n_jobs=-1
    )
    
    randomized_search_xgb.fit(X_train, y_train)
    best_params_xgb = randomized_search_xgb.best_params_
    print("Best parameters are: ", best_params_xgb)

    # Get the best model
    best_model = randomized_search_xgb.best_estimator_

    # Evaluate the model
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Training score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    print(f"MAE on training set: {mae_train:.2f}")
    print(f"MAE on test set: {mae_test:.2f}")

    # Save the best model
    with open('models/xgb_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("XGBoost model saved in 'models/xgb_model.pkl'")

def main():
    # Define the columns to keep
    columns_to_keep = [
        "BathroomCount", "BedroomCount", "ConstructionYear", "District",
        "Garden", "GardenArea", "Kitchen", "LivingArea", "Locality",
        "NumberOfFacades", "Price", "Province", "Region", "RoomCount",
        "ShowerCount", "SubtypeOfProperty", "SurfaceOfPlot", "SwimmingPool",
        "Terrace", "PEB_Encoded", "State_Encoded", "FloodingZone_Encoded", "Furnished"
    ]
    
    # Load the data
    df_sales = load_data("data/dataset_sales_cleaned.csv", columns_to_keep)
    
    if df_sales is not None:
        # Preprocess the entire data to ensure consistent dummy variable columns
        df_sales_dummies = preprocess_data_with_dummies(df_sales)
        
        # Save unique categorical values
        save_preprocessor_and_unique_values(df_sales)
        
        # Split the data into training and testing sets
        X = df_sales_dummies.drop('Price', axis=1)
        y = df_sales_dummies['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
        
        # Train and evaluate the model
        train_and_evaluate_xgboost(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
