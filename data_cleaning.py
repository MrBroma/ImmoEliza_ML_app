import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_data():
    data = pd.read_json("data/final_dataset.json")
    postal_codes_to_keep = pd.read_csv('data/postalcode_be.csv')
    postal_codes_to_keep_list = postal_codes_to_keep['Code postal'].tolist()
    data = data[data['PostalCode'].isin(postal_codes_to_keep_list)]
    return data

def handle_outliers(data, columns, factor=1.5):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    return data


def clean_data(data):
    data.drop_duplicates("PropertyId", inplace=True)
    data = data.dropna(subset=['District', 'Province', 'Region', 'Locality'])
    data = data[~data.TypeOfSale.isin(["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"])]
    return data


def split_data(data):
    data_sales = data[data.TypeOfSale == "residential_sale"]
    data_rent = data[data.TypeOfSale == "residential_monthly_rent"]
    return data_sales, data_rent

def preprocess_sales_data(data_sales):
    data_sales.drop(['MonthlyCharges'], axis=1, inplace=True)
    
    # fillna to 0
    columns_to_fill = ['Fireplace', 'Garden', 'Furnished', 'SwimmingPool', 'ShowerCount', 'Terrace', 'ToiletCount']
    data_sales[columns_to_fill] = data_sales[columns_to_fill].fillna(0)
    data_sales['NumberOfFacades'].fillna(2, inplace=True)

    # fill na to unknown etc...
    data_sales['FloodingZone'].fillna('NON_FLOOD_ZONE', inplace=True)
    data_sales['Kitchen'].fillna('NOT_INSTALLED', inplace=True)
    data_sales['StateOfBuilding'].fillna('GOOD', inplace=True)
    data_sales['PEB'].fillna('Unknown', inplace=True)

    # fill with median
    # columns_to_fill_median = ['GardenArea', 'SurfaceOfPlot', 'RoomCount', 'ConstructionYear', 'BathroomCount']
    # data_sales[columns_to_fill_median] = data_sales[columns_to_fill_median].apply(lambda x: x.fillna(x.median()))


    keep_PEB = ['A++', 'A+', 'B', 'C', 'D', 'E', 'F', 'G']
    data_sales = data_sales[data_sales['PEB'].isin(keep_PEB)]
    
    # data modif to 0 or 1
    data_sales['FloodingZone'] = data_sales['FloodingZone'].apply(lambda zone: 0 if zone == 'NON_FLOOD_ZONE' else 1)
    data_sales['LivingArea'].fillna(data_sales['LivingArea'].median(), inplace=True)
    data_sales = data_sales.drop(columns=['Url', 'Country', 'TypeOfSale', 'PropertyId', 'TypeOfProperty', 'PostalCode'])

    # Modify column to int
    columns_to_int64 = [
    'BathroomCount', 'Fireplace', 'Furnished', 'Garden', 'NumberOfFacades',
    'RoomCount', 'ShowerCount', 'SurfaceOfPlot', 'SwimmingPool', 'Terrace',
    'ToiletCount', 'LivingArea', 'ConstructionYear', 'GardenArea', 'BedroomCount'
    ]
    data_sales[columns_to_int64] = data_sales[columns_to_int64].astype('int64')
    
    data_sales['Locality'] = data_sales['Locality'].astype(str).str.upper()

    # data to strip
    columns_to_strip = ['Province', 'District', 'Region', 'SubtypeOfProperty']
    data_sales[columns_to_strip] = data_sales[columns_to_strip].astype(str).str.upper().str.replace(' ', '_')

    return data_sales


loading = load_data()
columns = ['BathroomCount', 'BedroomCount', 'ConstructionYear', 'GardenArea', 'LivingArea', 'NumberOfFacades',
            'Price', 'RoomCount', 'ShowerCount', 'SurfaceOfPlot', 'ToiletCount']

outliers = handle_outliers(loading, columns)
cleaning = clean_data(outliers)
data_sales, data_rent = split_data(cleaning)
data_cleaned = preprocess_sales_data(data_sales)
data_cleaned.reset_index(drop=True, inplace=True)


print(data_cleaned.head())
print(data_cleaned.shape)

data_cleaned.to_csv('data/dataset_sales_cleaned.csv', index=False)

