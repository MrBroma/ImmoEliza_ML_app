import streamlit as st
import pandas as pd
import pickle


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("House Price Prediction")

    st.sidebar.header("Input Features")

    # Extract features
    district_options = df['District'].unique().tolist()
    kitchen_options = df['Kitchen'].unique().tolist()
    locality_options = df['Locality'].unique().tolist()
    province_options = df['Province'].unique().tolist()
    region_options = df['Region'].unique().tolist()
    subtype_options = df['SubtypeOfProperty'].unique().tolist()
    peb_options = df['PEB'].unique().tolist()
    state_options = df['StateOfBuilding'].unique().tolist()
    flooding_zone_options = df['FloodingZone'].unique().tolist()

    # Input features
    bedroom_count = st.sidebar.number_input("Number of Bedrooms", min_value=0, value=int(df['BedroomCount'].mean()))
    bathroom_count = st.sidebar.number_input("Number of Bathrooms", min_value=0, value=int(df['BathroomCount'].mean()))
    construction_year = st.sidebar.number_input(
        "Construction Year", min_value=1900, max_value=2032, value=int(df['ConstructionYear'].median()))
    district = st.sidebar.selectbox("District", district_options)
    garden = st.sidebar.selectbox("Garden", ["Yes", "No"])
    garden_area = st.sidebar.number_input("Garden Area (in m²)", min_value=0.0, value=df['GardenArea'].mean())
    kitchen = st.sidebar.selectbox("Kitchen Type", kitchen_options)
    living_area = st.sidebar.number_input("Living Area (in m²)", min_value=0.0, value=df['LivingArea'].mean())
    locality = st.sidebar.selectbox("Locality", locality_options)
    number_of_facades = st.sidebar.number_input("Number of Facades", min_value=0, value=int(df['NumberOfFacades'].mean()))
    province = st.sidebar.selectbox("Province", province_options)
    region = st.sidebar.selectbox("Region", region_options)
    room_count = st.sidebar.number_input("Number of Rooms", min_value=0, value=int(df['RoomCount'].mean()))
    shower_count = st.sidebar.number_input("Number of Showers", min_value=0, value=int(df['ShowerCount'].mean()))
    subtype_of_property = st.sidebar.selectbox("Subtype of Property", subtype_options)
    surface_of_plot = st.sidebar.number_input("Surface of Plot (in m²)", min_value=0.0, value=df['SurfaceOfPlot'].mean())
    swimming_pool = st.sidebar.selectbox("Swimming Pool", ["Yes", "No"])
    terrace = st.sidebar.selectbox("Terrace", ["Yes", "No"])
    peb = st.sidebar.selectbox("PEB", peb_options)
    stateofbuilding = st.sidebar.selectbox("State of Building", state_options)
    flooding_zone = st.sidebar.selectbox("Flooding Zone", flooding_zone_options)
    furnished = st.sidebar.selectbox("Furnished", ["Yes", "No"])

    # Prepare input data
    input_data = pd.DataFrame({
        'BedroomCount': [bedroom_count],
        'BathroomCount': [bathroom_count],
        'ConstructionYear': [construction_year],
        'District': [district],
        'Garden': [garden],
        'GardenArea': [garden_area],
        'Kitchen': [kitchen],
        'LivingArea': [living_area],
        'Locality': [locality],
        'NumberOfFacades': [number_of_facades],
        'Province': [province],
        'Region': [region],
        'RoomCount': [room_count],
        'ShowerCount': [shower_count],
        'SubtypeOfProperty': [subtype_of_property],
        'SurfaceOfPlot': [surface_of_plot],
        'SwimmingPool': [swimming_pool],
        'Terrace': [terrace],
        'PEB': [peb],
        'StateOfBuilding': [stateofbuilding],
        'FloodingZone': [flooding_zone],
        'Furnished': [furnished]
    })

    if st.sidebar.button("Predict Price"):
        # Transform input data using the preprocessor
        transformed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(transformed_data)[0]
        
        # Show results
        st.markdown(f"<h2 class='header'>Predicted Price: {prediction:,.2f} €</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Load the preprocessor and model
    preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
    model = pickle.load(open("models/random_forest.pkl", "rb"))

    # Load the dataset to retrieve unique values for dropdowns
    df = pd.read_csv("data/dataset_sales_cleaned.csv")
    main()

