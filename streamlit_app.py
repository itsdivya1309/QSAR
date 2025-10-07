# streamlit_app.py
import pandas as pd
import streamlit as st
import joblib
import os
import py3Dmol
import streamlit.components.v1 as components
from utils import *
from openbabel import pybel
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="QSAR Model", layout="wide")

# Sidebar navigation (vertical)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Predict", "Visualisation / About"])

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
integrase_models = {name: load_model(p) for name, p in INTEGRAE_MODELS.items()}


# -----------------------------------------------------------------------
# Page: Home / Predict
# -----------------------------------------------------------------------
def page_predict():
    st.title("Home - Predict pIC50 of the compound.")
    st.markdown("Enter a SMILES string and choose the target - Protease or Integrase. The application would compute molecular descriptors and predict pIC50 for the molecule. We also check the Applicability Domain (AD) and show the 3D view of the molecule.")

    col_left, col_right = st.columns([0.30, 0.70], gap='large')

    with col_left:
        st.subheader("Input SMILES")
        smiles = st.text_area("Enter the SMILES here", 
                              value="CC(=O)Oc1ccccc1C(=O)O", 
                              height=120)
        target = st.selectbox("Target", ['Protease Inhibitor', 'Integrase Inhibitor'])
        run = st.button("Predict pIC50")

        if run:
            if not smiles.strip():
                st.error("Please enter a SMILES string.")
                return
            
            # Compute features
            try:
                x = build_feature_df(smiles, morgan_radius=2, nBits=512)
                st.success("Features generated.")
            except Exception as e:
                st.error(f"Feature generation failed: {e}")
                return
            
            # Load expected training fingerprints to check AD
            try: 
                if target.startswith("Protease"):
                    X_train = pd.read_csv('data/pi_qsar_features.csv')
                    expected_cols = pd.read_csv('artifacts/protease_features.csv')
                    models = protease_models
                else:
                    X_train = pd.read_csv('data/ii_qsar_features.csv')
                    expected_cols = pd.read_csv('artifacts/integrase_features.csv')
                    models = integrase_models
            except Exception as e:
                st.warning(f"Could not load training fingerprints for AD check: {e}")
                models = protease_models if target.startswith("Protease") else integrase_models

            # AD Check (kNN / Jaccard on fingerprints)
            if X_train is not None:
                try:
                    in_ad, avg_dist = check_applicability_knn(X_train, x)
                    if in_ad:
                        st.success(f"Within Applicability Domain (avg Jaccard dist = {avg_dist:.3f})")
                    else:
                        st.warning(f"Outside Applicability Domain (avg Jaccard dist = {avg_dist:.3f}) — treat predictions with caution.")
                except Exception as e:
                    st.warning(f'AD check failed: {e}')

            # Make prediction
            try:
                expected_cols = expected_cols.iloc[:,0].tolist()
                x_final = select_actual_columns(x, expected_cols)
                pred = make_prediction(x_final, models)
                st.metric(label="Predicted pIC50", value=f"{float(pred):.3f}")
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    with col_right:
        st.subheader('3D viewer')
        if not smiles.strip():
            st.info("Enter a SMILES on the left to render a 3D structure.")
        else:
            # Try rendering 3D viewer HTML from utils.smiles_to_3d_view
            try:
                html = smiles_to_3d_view(smiles, width=700, height=500)
                # Embed the returned HTML from py3Dmol
                st.components.v1.html(html, height=520, scrolling=True)
            except Exception as e_3d:
                st.warning(f"3D rendering failed: {e_3d}. Falling back to PubChem 2D image.")

                # Fallback: ask PubChem for a 2D PNG
                try:
                    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/PNG"
                    resp = requests.get(pubchem_url, timeout=10)
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content))
                        st.image(img, caption="2D structure from PubChem", use_column_width=True)
                    else:
                        st.error("PubChem fallback failed (SMILES might be invalid or not found).")
                except Exception as e_fallback:
                    st.error(f"PubChem fallback also failed: {e_fallback}")
            
# -----------------------------------------------------------------------
# Page: Visualisation / About
# -----------------------------------------------------------------------
def page_visuals():
    st.title("Visualisation & About")
    st.markdown("Model diagnostics, training distribution, AD diagnostics and project info.")

    st.subheader("About this project")
    st.markdown("""
    ### Goal: 
    Predict pIC50 for Protease / Integrase inhibitors of HIV-1.
                
    ### AD Check:
    kNN Jaccard on Morgan fingerprints (default k=5, threshold=0.25)
                
    ### Models:
    SVM, RF, kNN, Stacking (ensembler).           
    """)

    st.subheader("Training set overview")
    tab1, tab2 = st.tabs(["Protease", "Integrase"])
    with tab1:
        try:
            df = pd.read_csv('data/pi_df_clean.csv')  # full -> return descriptors + target
            st.markdown("**Protease — summary**")
            st.write(df.describe())
            st.markdown("### pIC50 distribution")

            williams_path = "artifacts/protease_williams.png"
            tanimoto_path = "artifacts/protease_tanimoto.png"

            if st.button("Show Protease Diagnostics"):
                col_w, col_t = st.columns([1, 1], gap="large")
                with col_w:
                    st.image(williams_path, caption="Protease — Williams plot", use_column_width=True)

                with col_t:
                    st.image(tanimoto_path, caption="Protease — Tanimoto similarity heatmap", use_column_width=True)

        except Exception as e:
            st.warning(f"Could not load protease training data: {e}")

    with tab2:
        try:
            df = pd.read_csv('data/ii_df_clean.csv')
            st.markdown("**Integrase — summary**")
            st.write(df.describe())
            st.markdown("### pIC50 distribution")
            
            williams_path = "artifacts/integrase_williams.png"
            tanimoto_path = "artifacts/integrase_tanimoto.png"

            if st.button("Show Integrase Diagnostics"):
                col_w, col_t = st.columns([1, 1], gap="large")
                with col_w:
                    st.image(williams_path, caption="Integrase — Williams plot", use_column_width=True)

                with col_t:
                    st.image(tanimoto_path, caption="Integrase — Tanimoto similarity heatmap", use_column_width=True)

        except Exception as e:
            st.warning(f"Could not load integrase training data: {e}")

    st.markdown("""
                **Submitted by:** Divya Tiwari

                **Roll no.:** 21014009
                
                **Branch:** Biochemical Engineering""")






# ---------- Router ----------
if page == "Home / Predict":
    page_predict()
else:
    page_visuals()