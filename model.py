import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import uniform, randint
import pickle
import json
import os

def load_data(columns_to_keep):
    try:
        data = pd.read_csv("data/dataset_sales_cleaned.csv")
        data_sales = data[columns_to_keep]
        return data_sales
    except FileNotFoundError:
        print("Le fichier spécifié est introuvable.")
        return None

def prepare_data(data_sales):
    ode_cols = ["State_Encoded", "PEB_Encoded"]
    ohe_cols = ["District", "Locality", "Province", "Region",
                "SubtypeOfProperty", "Kitchen", "Garden",
                "SwimmingPool", "Terrace", "Furnished"]
    num_cols = ["BathroomCount", "BedroomCount", "ConstructionYear", "GardenArea",
                "LivingArea", "NumberOfFacades", "RoomCount", "ShowerCount", "SurfaceOfPlot", "FloodingZone_Encoded"]
    
    num_pipeline = Pipeline(steps=[
        ('impute', KNNImputer()),
        ('scaler', StandardScaler())
    ])
    
    ode_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler()) 
    ])
    
    ohe_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    col_trans = ColumnTransformer(transformers=[
        ('num_p', num_pipeline, num_cols),
        ('ode_p', ode_pipeline, ode_cols),
        ('ohe_p', ohe_pipeline, ohe_cols)
    ], remainder='passthrough', n_jobs=-1)
    
    return col_trans

def save_preprocessor_and_unique_values(data_sales, col_trans):
    if not os.path.exists('models'):
        os.makedirs('models')
    
    X = data_sales.drop('Price', axis=1)
    col_trans.fit(X)  # Ajuster le préprocesseur
    
    # Sauvegarder le ColumnTransformer
    with open('models/preprocessor.pkl', 'wb') as file:
        pickle.dump(col_trans, file)
    print("Préprocesseur sauvegardé dans 'models/preprocessor.pkl'")
    
    # Extraire et sauvegarder les valeurs uniques pour les caractéristiques catégoriques
    unique_values = {}
    for column in data_sales.select_dtypes(include=['object']).columns:
        unique_values[column] = data_sales[column].unique().tolist()
    
    with open('models/unique_values.json', 'w') as file:
        json.dump(unique_values, file)
    print("Valeurs uniques sauvegardées dans 'models/unique_values.json'")

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, col_trans):
    xgb_model = XGBRegressor(tree_method='gpu_hist', use_label_encoder=False, eval_metric='rmse')
    
    param_distributions_xgb = {
        'preprocessing__num_p__impute__n_neighbors': [3, 5, 7, 9],
        'model__n_estimators': randint(50, 200),
        'model__max_depth': randint(3, 10),
        'model__learning_rate': uniform(0.01, 0.3),
    }
    
    pipeline_xgb = Pipeline(steps=[
        ('preprocessing', col_trans),
        ('model', xgb_model)
    ])
    
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
    print("Meilleurs paramètres pour XGBoost: ", best_params_xgb)

    # Évaluer le modèle
    best_model = randomized_search_xgb.best_estimator_

    # Prédictions sur l'ensemble d'entraînement et de test
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calcul des métriques
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Score d'entraînement : {train_score:.4f}")
    print(f"Score de test : {test_score:.4f}")
    print(f"MAE sur l'ensemble d'entraînement : {mae_train:.2f}")
    print(f"MAE sur l'ensemble de test : {mae_test:.2f}")

    # Sauvegarder le pipeline entier, y compris le préprocesseur ajusté
    with open('models/xgb_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Modèle XGBoost sauvegardé dans 'models/xgb_model.pkl'")


def main():
    columns_to_keep = [
        "BathroomCount", "BedroomCount", "ConstructionYear", "District",
        "Garden", "GardenArea", "Kitchen", "LivingArea", "Locality",
        "NumberOfFacades", "Price", "Province", "Region", "RoomCount",
        "ShowerCount", "SubtypeOfProperty", "SurfaceOfPlot", "SwimmingPool",
        "Terrace", "PEB_Encoded", "State_Encoded", "FloodingZone_Encoded", "Furnished"
    ]
    
    df_sales = load_data(columns_to_keep)
    
    if df_sales is not None:
        col_trans = prepare_data(df_sales)
        save_preprocessor_and_unique_values(df_sales, col_trans)  # Sauvegarder le préprocesseur et les valeurs uniques
        
        X = df_sales.drop('Price', axis=1)
        y = df_sales['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
        
        # Entraîner et évaluer le modèle XGBoost
        train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, col_trans)

        
if __name__ == "__main__":
    main()
