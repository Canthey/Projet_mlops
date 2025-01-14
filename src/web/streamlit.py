import streamlit as st
import requests

st.title("ImmoPrix - Prédiction avec l'API MLOps")

med_inc = st.slider("Revenu médian du quartier (MedInc, en dizaines de milliers de $)", min_value=0.0, max_value=15.0, value=0.0, step=0.1)
house_age = st.slider("Âge médian des maisons dans le quartier (HouseAge, en années)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
ave_rooms = st.slider("Nombre moyen de pièces par logement (AveRooms)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
ave_bedrooms = st.slider("Nombre moyen de chambres par logement (AveBedrms)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
population = st.slider("Population totale du quartier (Population)", min_value=0, max_value=50000, value=0, step=100)
ave_occupation = st.slider("Nombre moyen d'occupants par logement (AveOccup)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
latitude = st.slider("Latitude géographique du quartier (Latitude)", min_value=32.0, max_value=42.0, value=32.0, step=0.1)
longitude = st.slider("Longitude géographique du quartier (Longitude)", min_value=-125.0, max_value=-114.0, value=-125.0, step=0.1)

if st.button("Prédire"):
    data = {
        "med_inc": med_inc,
        "house_age": house_age,
        "ave_rooms": ave_rooms,
        "ave_bedrooms": ave_bedrooms,
        "population": population,
        "ave_occupation": ave_occupation,
        "latitude": latitude,
        "longitude": longitude
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            headers={"accept": "application/json", "Content-Type": "application/json"},
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Score prédit : {result}")
        else:
            st.error(f"Erreur {response.status_code} : {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la requête : {e}")
