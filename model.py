import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error


def load_data(columns_to_keep):
    data = pd.read_csv("data/dataset_sales_cleaned.csv")
    data_sales = data[columns_to_keep]
    return data_sales

def prepare_data(data_sales):
    # List for ordinal encoding
    ode_cols = ["State_Encoded", "PEB_Encoded"]
    # Listfor one-hot encoding
    ohe_cols = ["District", "Locality", "Province",
                "Region", "SubtypeOfProperty", "Kitchen", "Garden",
                "SwimmingPool", "Terrace"]
    # List num colomn
    num_cols = ["BathroomCount", "BedroomCount", "ConstructionYear", "GardenArea",
                "LivingArea", "NumberOfFacades", "RoomCount", "ShowerCount", "SurfaceOfPlot", "FloodingZone_Encoded"]
    # num pipeline
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

def train_model(data_sales, col_trans):
    X = df_sales.drop('Price', axis=1)
    y = df_sales['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    pipeline = Pipeline(steps=[
        ('preprocessing', col_trans),
        ('model', GradientBoostingRegressor())
    ])

    param_grid = {
        'preprocessing__num_p__impute__n_neighbors': [3, 5, 7, 9, 11]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['preprocessing__num_p__impute__n_neighbors']
    
    pipeline.set_params(preprocessing__num_p__impute__n_neighbors=best_k)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    return train_score, test_score, mae


# variables
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

# Lauch functions
df_sales = load_data(columns_to_keep)
data_preparation = prepare_data(df_sales)

train_score, test_score, mae = train_model(df_sales, data_preparation)

print("Train Score: ", train_score)
print("Test Score: ", test_score)
print("Mean Absolute Error: ", mae)

