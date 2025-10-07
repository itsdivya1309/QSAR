# streamlit_app.py
import streamlit as st
import joblib
import os
# from utils import mol_from_smiles, mol_to_2d_image, mol_to_3d_pdb, prepare_features_for_model, pil_image_to_bytes
import streamlit.components.v1 as components

st.set_page_config(page_title="SMILES Predictor", layout="centered")


MODEL_DIR = "models"
PROTEASE_MODELS = {
    "svm": os.path.join(MODEL_DIR, "pi_svm.joblib"),
    "knn": os.path.join(MODEL_DIR, "pi_knn.joblib"),
    "rf": os.path.join(MODEL_DIR, "pi_rf.joblib"),
    "ensembler": os.path.join(MODEL_DIR, "pi_meta.joblib")
}
INTEGRAE_MODELS = {
    "svm": os.path.join(MODEL_DIR, "ii_svm.joblib"),
    "knn": os.path.join(MODEL_DIR, "ii_knn.joblib"),
    "rf": os.path.join(MODEL_DIR, "ii_rf.joblib"),
    "ensembler": os.path.join(MODEL_DIR, "ii_meta.joblib")
}

@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# load models lazily
protease_models = {name: load_model(p) for name, p in PROTEASE_MODELS.items()}
# integrase_models = {name: load_model(p) for name, p in INTEGRAE_MODELS.items()}


st.title("SMILES → Structure + Model Prediction")
st.write("Enter a SMILES string and choose which model to use (Protease or Integrase).")
