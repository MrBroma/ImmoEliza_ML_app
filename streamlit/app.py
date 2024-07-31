import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.exceptions import NotFittedError

# Chargement du modèle et du préprocesseur
@st.cache_resource
def load_model():
    try:
        with open('models/xgb_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Le fichier du modèle n'a pas été trouvé.")
    except pickle.UnpicklingError:
        st.error("Erreur lors du chargement du modèle.")
    return None

@st.cache_resource
def load_preprocessor():
    try:
        with open('models/preprocessor.pkl', 'rb') as file:
            preprocessor = pickle.load(file)
        return preprocessor
    except FileNotFoundError:
        st.error("Le fichier du préprocesseur n'a pas été trouvé.")
    except pickle.UnpicklingError:
        st.error("Erreur lors du chargement du préprocesseur.")
    return None

@st.cache_resource
def load_unique_values():
    try:
        with open('models/unique_values.json', 'r') as file:
            unique_values = json.load(file)
        return unique_values
    except FileNotFoundError:
        st.error("Le fichier des valeurs uniques n'a pas été trouvé.")
    except json.JSONDecodeError:
        st.error("Erreur lors du chargement des valeurs uniques.")
    return {}

# Chargement des ressources
model = load_model()
preprocessor = load_preprocessor()
unique_values = load_unique_values()

def predict_price(features):
    if preprocessor is None:
        st.error("Le préprocesseur n'est pas chargé correctement.")
        return None

    # Liste des colonnes attendues par le préprocesseur
    all_columns = [
        "BathroomCount", "BedroomCount", "ConstructionYear", "District", "Garden",
        "GardenArea", "Kitchen", "LivingArea", "Locality", "NumberOfFacades",
        "PEB_Encoded", "State_Encoded", "FloodingZone_Encoded", "SurfaceOfPlot",
        "SwimmingPool", "Terrace", "Furnished", "RoomCount", "ShowerCount", "ToiletCount",
        "Region", "Province", "SubtypeOfProperty"
    ]

    # Créer un DataFrame avec toutes les colonnes nécessaires
    features_df = pd.DataFrame([features], columns=all_columns).fillna(0)  # Remplir les valeurs manquantes avec 0 ou une autre valeur par défaut

    try:
        # Transformer les données d'entrée
        processed_data = preprocessor.transform(features_df)
        # Faire la prédiction
        prediction = model.predict(processed_data)
        return prediction[0]
    except NotFittedError:
        st.error("Le préprocesseur n'a pas été ajusté. Veuillez vérifier les fichiers sauvegardés.")
    except Exception as e:
        st.error(f"Erreur lors de la transformation des données : {str(e)}")
    return None

# Interface utilisateur Streamlit
def main():
    st.title("Prédiction du Prix Immobilier")
    st.sidebar.header("Entrer les caractéristiques du bien")

    # Créer les champs pour chaque caractéristique avec les options dynamiques
    bathroom_count = st.sidebar.slider("Nombre de Salles de Bain", 0, 10, 1)
    bedroom_count = st.sidebar.slider("Nombre de Chambres", 0, 10, 1)
    construction_year = st.sidebar.number_input("Année de Construction", 1900, 2024, 2000)
    
    district = st.sidebar.selectbox("District", unique_values.get("District", ["Inconnu"]))
    garden = st.sidebar.selectbox("Jardin", unique_values.get("Garden", ["0", "1"]))
    garden_area = st.sidebar.number_input("Superficie du Jardin (m²)", 0, 10000, 0)
    kitchen = st.sidebar.selectbox("Cuisine", unique_values.get("Kitchen", ["NOT_INSTALLED", "SEMI_EQUIPPED", "INSTALLED", "HYPER_EQUIPPED"]))
    living_area = st.sidebar.number_input("Surface Habitable (m²)", 0, 10000, 100)
    locality = st.sidebar.selectbox("Localité", unique_values.get("Locality", ["Inconnu"]))
    number_of_facades = st.sidebar.slider("Nombre de Façades", 0, 10, 1)
    peb_encoded = st.sidebar.selectbox("PEB", unique_values.get("PEB_Encoded", ["A", "B", "C", "D", "E", "F"]))
    state_encoded = st.sidebar.selectbox("État du Bâtiment", unique_values.get("State_Encoded", ["AS_NEW", "GOOD", "TO_BE_DONE_UP", "TO_RESTORE", "TO_RENOVATE"]))
    flooding_zone_encoded = st.sidebar.selectbox("Zone d'Inondation", unique_values.get("FloodingZone_Encoded", ["NON_FLOOD_ZONE", "POSSIBLE_FLOOD_ZONE"]))
    surface_of_plot = st.sidebar.number_input("Surface du Terrain (m²)", 0, 10000, 0)
    swimming_pool = st.sidebar.selectbox("Piscine", unique_values.get("SwimmingPool", ["0", "1"]))
    terrace = st.sidebar.selectbox("Terrasse", unique_values.get("Terrace", ["0", "1"]))
    furnished = st.sidebar.selectbox("Meublé", unique_values.get("Furnished", ["0", "1"]))
    room_count = st.sidebar.slider("Nombre de Pièces", 1, 20, 1)
    shower_count = st.sidebar.slider("Nombre de Douche", 0, 10, 1)
    toilet_count = st.sidebar.slider("Nombre de Toilettes", 0, 10, 1)

    features = {
        "BathroomCount": bathroom_count,
        "BedroomCount": bedroom_count,
        "ConstructionYear": construction_year,
        "District": district,
        "Garden": garden,
        "GardenArea": garden_area,
        "Kitchen": kitchen,
        "LivingArea": living_area,
        "Locality": locality,
        "NumberOfFacades": number_of_facades,
        "PEB_Encoded": peb_encoded,
        "State_Encoded": state_encoded,
        "FloodingZone_Encoded": flooding_zone_encoded,
        "SurfaceOfPlot": surface_of_plot,
        "SwimmingPool": swimming_pool,
        "Terrace": terrace,
        "Furnished": furnished,
        "RoomCount": room_count,
        "ShowerCount": shower_count,
        "ToiletCount": toilet_count,
        "Region": "Inconnu",  # Valeur par défaut
        "Province": "Inconnu",  # Valeur par défaut
        "SubtypeOfProperty": "Inconnu"  # Valeur par défaut
    }

    if st.sidebar.button("Prédire le Prix"):
        # Obtenir la prédiction
        price = predict_price(features)
        if price is not None:
            st.write(f"**Prix prédit :** {price:,.2f} EUR")

if __name__ == "__main__":
    main()
