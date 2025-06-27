import streamlit as st
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model and label names
with open("tox_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_cols.pkl", "rb") as f:
    label_cols = pickle.load(f)

# Endpoint descriptions
endpoint_descriptions = {
    "NR-AR": "Androgen receptor activation (hormonal disruption).",
    "NR-AR-LBD": "Androgen receptor ligand binding domain interaction.",
    "NR-AhR": "Aryl hydrocarbon receptor activation (often pollutant-related toxicity).",
    "NR-Aromatase": "Inhibition of aromatase enzyme (affects estrogen synthesis).",
    "NR-ER": "Estrogen receptor activation (hormonal disruption).",
    "NR-ER-LBD": "Estrogen receptor ligand binding domain interaction.",
    "NR-PPAR-gamma": "PPAR-gamma nuclear receptor activity (metabolism-related).",
    "SR-ARE": "Oxidative stress response (via antioxidant response element).",
    "SR-ATAD5": "DNA damage and repair stress response (via ATAD5).",
    "SR-HSE": "Heat shock stress pathway activation.",
    "SR-MMP": "Mitochondrial membrane potential disruption.",
    "SR-p53": "p53 tumor suppressor pathway activation (DNA damage response)."
}

# ECFP4 fingerprint function
def smiles_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return np.array(fp)

# Streamlit UI
st.title("🧪 Toxicity Prediction App")

# Brief description
st.markdown(
    """
    This tool predicts the potential toxicity of a compound based on its **SMILES** (Simplified Molecular Input Line Entry System) representation.  
    It uses a machine learning model trained on the **Tox21 dataset**, which includes 12 different biological toxicity endpoints.  
    Each prediction score represents the **probability** that the molecule will show toxic activity in a specific biological assay.
    """
)

# SMILES input section
st.markdown("### 💊 Enter a SMILES string to predict toxicity:")
smiles = st.text_input("Canonical SMILES", placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)")

# If user entered a SMILES string
if smiles:
    fp = smiles_to_ecfp(smiles)
    if fp is None:
        st.error("❌ Invalid SMILES. Please enter a valid molecule.")
    else:
        # Make predictions
        probs = model.predict_proba([fp])
        preds = [p[0][1] for p in probs]
        result = dict(zip(label_cols, preds))

        # Display results
        st.subheader("🧬 Toxicity Predictions")
        for label, score in result.items():
            st.write(f"**{label}**: {score:.3f}")
        
        st.bar_chart(result)

        # Explanation of scores
        st.markdown(
            """
            ### 📊 How to Read the Scores  
            Each score ranges from **0.00 to 1.00**.  
            - A **higher score** (closer to 1) means the molecule is **more likely to be toxic** on that endpoint.  
            - A **lower score** (closer to 0) means it is **less likely to be toxic**.
            """
        )

        # Expandable endpoint descriptions
        with st.expander("📘 Click to show endpoint descriptions"):
            for label in label_cols:
                st.markdown(f"**{label}**: {endpoint_descriptions[label]}")
