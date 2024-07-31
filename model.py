import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error


def load_data():
    data = pd.read_csv("data/dataset_sales_cleaned.csv")
    return data


def prepare_data(data_sales, columns):
    data_sales = data_sales[columns_to_keep]








# variables
columns_to_keep = [
        'BathroomCount', 'BedroomCount', 'LivingArea', 'SurfaceOfPlot', 'ToiletCount', 'State_Encoded',
        'District', 'FloodingZone_Encoded', 'Kitchen', 'Province', 'Region', 'SubtypeOfProperty', 'PEB_Encoded', 
        'Garden', 'NumberOfFacades', 'SwimmingPool', 'Price', 'Locality'
    ]
# Lauch functions
df_sales = load_data()
data_preparation = prepare_data(df_sales, columns_to_keep)


print(df_sales.head())