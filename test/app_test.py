import streamlit as st
import pandas as pd
import pickle
import json

# Preprocessor and model loading
@st.cache_data
def load_preprocessor_and_model():
    with open('models/preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    
    with open('models/xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('models/unique_values.json', 'r') as file:
        unique_values = json.load(file)
    
    return preprocessor, model, unique_values

preprocessor, model, unique_values = load_preprocessor_and_model()

# App title
st.title("House Price Prediction")

# User entry creation
def user_input_features():
    BathroomCount = st.number_input('Bathroom Number', min_value=0, value=1)
    BedroomCount = st.number_input('Bedroom Number', min_value=0, value=2)
    ConstructionYear = st.number_input('Construction Year', min_value=1900, value=2000)
    District = st.selectbox('District', unique_values.get('District', ['']))
    Garden = st.selectbox('Garden', ['Yes', 'No'])
    GardenArea = st.number_input('Garden Area (m²)', min_value=0, value=50)
    Kitchen = st.selectbox('Kitchen', ['Installed', 'Non installed'])
    LivingArea = st.number_input('Living Area (m²)', min_value=0, value=120)
    Locality = st.selectbox('Locality', unique_values.get('Locality', ['']))
    NumberOfFacades = st.number_input('Facade Number', min_value=0, value=2)
    Province = st.selectbox('Province', unique_values.get('Province', ['']))
    Region = st.selectbox('Région', unique_values.get('Region', ['']))
    RoomCount = st.number_input('Rooms', min_value=0, value=5)
    ShowerCount = st.number_input('Shower Number', min_value=0, value=2)
    SubtypeOfProperty = st.selectbox('Type of Property', ['House', 'Appartement', 'Other'])
    SurfaceOfPlot = st.number_input('Plot Surface (m²)', min_value=0, value=300)
    SwimmingPool = st.selectbox('Swimming Pool', ['Yes', 'No'])
    Terrace = st.selectbox('Terrace', ['Yes', 'No'])
    PEB_Encoded = st.selectbox('PEB class', ['A++', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown'])
    State_Encoded = st.selectbox('Building State', ['As new', 'Renovated', 'Good', 'To be done up', 'To renove', 'To restore', 'Unknown'])
    FloodingZone_Encoded = st.selectbox('Flooding Area', ['Non flooding Area', 'Flooding Area'])
    Furnished = st.selectbox('Furnished', ['Yes', 'No'])

    data = {
        'BathroomCount': BathroomCount,
        'BedroomCount': BedroomCount,
        'ConstructionYear': ConstructionYear,
        'District': District,
        'Garden': 1 if Garden == 'Yes' else 0,
        'GardenArea': GardenArea,
        'Kitchen': Kitchen,
        'LivingArea': LivingArea,
        'Locality': Locality,
        'NumberOfFacades': NumberOfFacades,
        'Province': Province,
        'Region': Region,
        'RoomCount': RoomCount,
        'ShowerCount': ShowerCount,
        'SubtypeOfProperty': SubtypeOfProperty,
        'SurfaceOfPlot': SurfaceOfPlot,
        'SwimmingPool': 1 if SwimmingPool == 'Yes' else 0,
        'Terrace': 1 if Terrace == 'Yes' else 0,
        'PEB_Encoded': PEB_Encoded,
        'State_Encoded': State_Encoded,
        'FloodingZone_Encoded': 1 if FloodingZone_Encoded == 'Flooding Area' else 0,
        'Furnished': 1 if Furnished == 'Yes' else 0
    }

    features = pd.DataFrame(data, index=[0])
    
    return features

# User data
df = user_input_features()

# Mapping categorical values to numerical
state_mapping = {'As new': 0, 'Renovated': 1, 'Good': 2, 'To be done up': 3, 'To renove': 4, 'To restore': 5, 'Unknown': 6}
df['State_Encoded'] = df['State_Encoded'].map(state_mapping)

peb_mapping = {'A++': 0, 'A+': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Unknown': 8}
df['PEB_Encoded'] = df['PEB_Encoded'].map(peb_mapping)

# Preprocessing
X = preprocessor.transform(df)

# Predict house price
predicted_price = model.predict(X)

# Display predicted price
st.subheader('Predicted House Price')
st.write(f"${predicted_price[0]:,.2f}")

