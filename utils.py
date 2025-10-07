# utils.py
from typing import Optional, Tuple, Dict, List, Sequence
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit import DataStructs
from openbabel import pybel
import py3Dmol

# -------------------------------------------------------------------
# 3D visualization of the molecule
# -------------------------------------------------------------------
def smiles_to_3d_view(smiles: str, width: int = 400, height: int = 400) -> str:
    """
    Convert a SMILES string to a 3D molecular viewer HTML using OpenBabel + py3Dmol.

    Args:
        smiles (str): SMILES string for the molecule.
        width (int): Width of the viewer in pixels.
        height (int): Height of the viewer in pixels.

    Returns:
        str: HTML content that can be rendered inside Streamlit.
    """
    # Convert SMILES to molecule
    mol = pybel.readstring("smi", smiles)
    mol.make3D()

    # Create 3Dmol viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol.write("mol"), "mol")
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()

    # Return embeddable HTML
    return viewer._make_html()


# -----------------------------------------------------------------------------
# Generate Molecular Descriptors and Morgan Fingerprints for the SMILES
# -----------------------------------------------------------------------------
DEFAULT_DESC_LIST = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'NumAromaticRings', 'HeavyAtomCount', 'FractionCSP3'
]

def mol_from_smiles_safe(smi: str) -> Optional[Chem.Mol]:
    """
    Return RDKit Mol for a SMILES string or None if parsing fails.
    Accepts None / NaN inputs safely.
    """
    if smi is None:
        return None
    # handle NaN floats
    if isinstance(smi, float) and np.isnan(smi):
        return None
    try:
        m = Chem.MolFromSmiles(str(smi))
        return m
    except Exception:
        return None
    
def compute_descriptors_for_mol(m: Chem.Mol) -> Dict[str, float]:
    """Compute a set of basic 2D descriptors for a single RDKit Mol."""
    if m is None:
        return {k: np.nan for k in DEFAULT_DESC_LIST}
    try:
        molwt = Descriptors.MolWt(m)
        logp, mr = rdMolDescriptors.CalcCrippenDescriptors(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        hbd = rdMolDescriptors.CalcNumHBD(m)
        hba = rdMolDescriptors.CalcNumHBA(m)
        rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        arom = rdMolDescriptors.CalcNumAromaticRings(m)
        hac = Descriptors.HeavyAtomCount(m)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(m)
        return {
            'MolWt': float(molwt),
            'MolLogP': float(logp),
            'MolMR': float(mr),
            'TPSA': float(tpsa),
            'NumHDonors': int(hbd),
            'NumHAcceptors': int(hba),
            'NumRotatableBonds': int(rot),
            'NumAromaticRings': int(arom),
            'HeavyAtomCount': int(hac),
            'FractionCSP3': float(fsp3)
        }
    except Exception:
        return {k: np.nan for k in DEFAULT_DESC_LIST}
    
def morgan_fp_array(m: Chem.Mol, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Return numpy array (0/1 int) of Morgan fingerprint for a single RDKit Mol."""
    if m is None:
        return np.zeros((nBits,), dtype=np.uint8)
    try:
        bitvect = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        return arr
    except Exception:
        return np.zeros((nBits,), dtype=np.uint8)
    
def fp_array_to_df(fp_arr: np.array, prefix: str = 'Morgan') -> pd.DataFrame:
    """Convert morgan fingerprint array into dataframe of shape (n_samples, nBits)"""
    nBits = fp_arr.shape[0]
    cols = [f'{prefix}_{i}' for i in range(nBits)]
    return pd.DataFrame([fp_arr], columns=cols)

# The actual function to be called
def build_feature_df(smiles: str, morgan_radius: int = 2, nBits: int = 1024) -> pd.DataFrame:
    '''For a given smiles, construct its feature dataframe having molecular descriptors and fingerprints.'''
    m = mol_from_smiles_safe(smiles)
    desc = compute_descriptors_for_mol(m)
    desc_df = pd.DataFrame([desc])
    morgan = morgan_fp_array(m, nBits=512)
    morgan_df = fp_array_to_df(morgan, prefix = f'Morgan_{nBits}')
    features = pd.concat([desc_df.reset_index(drop=True), morgan_df.reset_index(drop=True)], axis=1)
    return features

# ---------------------------------------------------------------------------
# Making prediction using the trained models
# ---------------------------------------------------------------------------
def select_actual_columns(x: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """
    Remove columns not used during training and ensure the same column order as expected_cols.
    """
    # Keep only the expected columns that actually exist in x
    actual_cols = [c for c in expected_cols if c in x.columns]
    # Select and reorder columns
    x_new = x[actual_cols].copy()
    return x_new

def make_prediction(f: pd.DataFrame, models: Dict[str, any]) -> float:
    """
    Predict pIC50 value for the given compund.
    """
    svm_out = models['svm'].predict(f)[0]
    rf_out = models['rf'].predict(f)[0]
    knn_out = models['knn'].predict(f)[0]
    meta_in = [[knn_out, svm_out, rf_out]]
    meta_out = models['ensembler'].predict(meta_in)
    return meta_out


# --------------------------------------------------------------------------
# Applicability Domain
# --------------------------------------------------------------------------
import re
from sklearn.metrics import pairwise_distances

def check_applicability_knn(
    X_train: pd.DataFrame,
    x_new: pd.DataFrame,
    k: int = 5,
    threshold: float = 0.25,
    fp_prefixes: str = 'Morgan'
) -> Tuple[bool, float]:
    """
    kNN Jaccard AD check using fingerprint columns.

    - X_train: DataFrame containing training data (may contain descriptors and fingerprint columns).
    - x_new: either a single-row DataFrame (preferred), a Series, or a 1D numpy array with the same fingerprint columns.
    - k: number of nearest neighbours to average.
    - threshold: maximum average Jaccard distance to be considered 'within AD' (default 0.25).
    - fp_prefixes: iterable of common prefixes used to recognise fingerprint columns.

    Returns:
      (in_ad_bool, avg_jaccard_distance)

    Notes:
    - The function looks for fingerprint columns by matching column names that start with any prefix
      in fp_prefixes (case-sensitive for prefixes provided; common prefixes included by default).
    - If fewer than k training rows exist, k is reduced to n_train.
    """
    X_train = X_train.drop(columns=['pIC50'], axis=1)

    # find fingerprint columns in X_train using prefixes (regex for safety)
    pattern = re.compile(rf"^({'|'.join([re.escape(p) for p in fp_prefixes])})")
    fp_cols = [c for c in X_train.columns if pattern.match(c)]

    if len(fp_cols) == 0:
        # no obvious fingerprint columns found — raise so user can decide what to do
        raise ValueError(
            "No fingerprint columns found in X_train. "
            f"Looked for prefixes: {tuple(fp_prefixes)}. "
            "If your Morgan columns use a different prefix, pass it via `fp_prefixes`."
        )

    # align x_row with fingerprint columns
    x_row = x_new.iloc[0]
    # Series or DataFrame row: ensure columns present
    missing = [c for c in fp_cols if c not in x_row.index]
    if missing:
        raise ValueError(f"x_new is missing fingerprint columns: {missing}")
    x_fp = x_row[fp_cols].values

    # prepare boolean arrays for Jaccard
    X_fp_matrix = X_train[fp_cols].astype(bool).values
    x_fp_vector = np.asarray(x_fp).astype(bool).reshape(1, -1)

    # adjust k if necessary
    n_train = X_fp_matrix.shape[0]
    if n_train == 0:
        raise ValueError("X_train contains zero rows.")
    if k > n_train:
        k = n_train

    # compute Jaccard distances
    dists = pairwise_distances(X_fp_matrix, x_fp_vector, metric="jaccard").flatten()
    nearest = np.sort(dists)[:k]
    avg = float(np.mean(nearest))

    return (avg <= threshold, avg)

def make_williams_plot(df: pd.DataFrame):
    """
    Simple Williams plot:
      - X: leverage (h)
      - Y: standardized residuals
    df should contain descriptor columns and 'pIC50' as target.
    """
    if 'pIC50' not in df.columns:
        raise ValueError("df must contain 'pIC50' column for Williams plot.")
    # split
    X = df.drop(columns=['pIC50']).values
    y = df['pIC50'].values
    # linear fit for residuals (ordinary least squares)
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])  # intercept
    beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
    y_pred = X_design @ beta
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    std_resid = residuals / np.sqrt(mse)
    # leverages
    H = X_design @ np.linalg.pinv(X_design.T @ X_design) @ X_design.T
    leverages = np.diag(H)
    h_star = 3 * (X_design.shape[1]) / X_design.shape[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(leverages, std_resid, alpha=0.7)
    ax.axvline(h_star, color='red', linestyle='--', label=f"h* = {h_star:.3f}")
    ax.axhline(3, color='orange', linestyle='--')
    ax.axhline(-3, color='orange', linestyle='--')
    ax.set_xlabel("Leverage (h)")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Williams plot")
    # Legend in top-right inside the plot
    ax.legend(loc='upper right', fontsize='small', framealpha=0.85)
    fig.tight_layout()
    return fig