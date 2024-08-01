import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np

# Chargement du préprocesseur et du modèle
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

# Titre de l'application
st.title("Prédiction du Prix d'une Maison")

# Création des entrées utilisateur
def user_input_features():
    BathroomCount = st.number_input('Nombre de salles de bain', min_value=0, value=1)
    BedroomCount = st.number_input('Nombre de chambres', min_value=0, value=2)
    ConstructionYear = st.number_input('Année de construction', min_value=1900, value=2000)
    District = st.selectbox('District', unique_values.get('District', ['']))
    Garden = st.selectbox('Jardin', ['Oui', 'Non'])
    GardenArea = st.number_input('Surface du jardin (m²)', min_value=0, value=50)
    Kitchen = st.selectbox('Cuisine', ['Installée', 'Non installée'])
    LivingArea = st.number_input('Surface habitable (m²)', min_value=0, value=120)
    Locality = st.selectbox('Localité', unique_values.get('Locality', ['']))
    NumberOfFacades = st.number_input('Nombre de façades', min_value=0, value=2)
    Province = st.selectbox('Province', unique_values.get('Province', ['']))
    Region = st.selectbox('Région', unique_values.get('Region', ['']))
    RoomCount = st.number_input('Nombre de pièces', min_value=0, value=5)
    ShowerCount = st.number_input('Nombre de douches', min_value=0, value=2)
    SubtypeOfProperty = st.selectbox('Type de propriété', ['Maison', 'Appartement', 'Autre'])
    SurfaceOfPlot = st.number_input('Surface du terrain (m²)', min_value=0, value=300)
    SwimmingPool = st.selectbox('Piscine', ['Oui', 'Non'])
    Terrace = st.selectbox('Terrasse', ['Oui', 'Non'])
    PEB_Encoded = st.selectbox('Classe PEB', ['A++', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'Inconnu'])
    State_Encoded = st.selectbox('État du bâtiment', ['Comme neuf', 'Rénové', 'Bon', 'À refaire', 'À rénover', 'À restaurer', 'Inconnu'])
    FloodingZone_Encoded = st.selectbox('Zone inondable', ['Non inondable', 'Inondable'])
    Furnished = st.selectbox('Meublé', ['Oui', 'Non'])
    
    data = {
        'BathroomCount': BathroomCount,
        'BedroomCount': BedroomCount,
        'ConstructionYear': ConstructionYear,
        'District': District,
        'Garden': 1 if Garden == 'Oui' else 0,
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
        'SwimmingPool': 1 if SwimmingPool == 'Oui' else 0,
        'Terrace': 1 if Terrace == 'Oui' else 0,
        'PEB_Encoded': PEB_Encoded,
        'State_Encoded': State_Encoded,
        'FloodingZone_Encoded': 1 if FloodingZone_Encoded == 'Inondable' else 0,
        'Furnished': 1 if Furnished == 'Oui' else 0
    }
    
    features = pd.DataFrame(data, index=[0])
    
    return features

# Collecte des données de l'utilisateur
user_data = user_input_features()

# Afficher les données entrées par l'utilisateur
st.write("Voici les caractéristiques de la maison que vous avez fournies :")
st.write(user_data)

# Prétraitement des données d'entrée
try:
    # Afficher les types de données avant transformation
    st.write("Types de données avant transformation :")
    st.write(user_data.dtypes)
    
    # Afficher les colonnes du préprocesseur
    st.write("Colonnes du préprocesseur :")
    st.write(preprocessor.get_feature_names_out())
    
    # Prétraitement manuel pour les variables catégorielles problématiques
    state_mapping = {'Comme neuf': 0, 'Rénové': 1, 'Bon': 2, 'À refaire': 3, 'À rénover': 4, 'À restaurer': 5, 'Inconnu': 6}
    user_data['State_Encoded'] = user_data['State_Encoded'].map(state_mapping)
    
    peb_mapping = {'A++': 0, 'A+': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'Inconnu': 8}
    user_data['PEB_Encoded'] = user_data['PEB_Encoded'].map(peb_mapping)

    # Appliquer le prétraitement
    X_processed = preprocessor.transform(user_data)
    
    # Afficher la forme et les premières lignes des données après transformation
    st.write("Données après prétraitement :")
    st.write(pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out()).head())
    
    # Prédiction
    prediction = model.predict(X_processed)
    st.write(f"Le prix prédit pour cette maison est : {prediction[0]:,.2f} €")
except Exception as e:
    st.error(f"Erreur lors de la transformation des données : {str(e)}")
    st.write("Détails de l'erreur :")
    st.exception(e)

