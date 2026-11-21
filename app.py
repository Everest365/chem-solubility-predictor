import urllib.parse
import urllib.request
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import streamlit as st

st.set_page_config(
    page_title="AI Solubility & Florida Aquifer Risk Screener", page_icon="🧪"
)
st.title("🧪 AI Chemical Solubility & Florida Aquifer Risk Screener")
st.write(
    "Predicts aqueous solubility (logS) using a Random Forest model (R² ="
    " 0.862, RMSE = 0.807) and evaluates environmental leaching and"
    " bioaccumulation risks."
)

model = joblib.load("solubility_model (2).pkl")


def name_to_smiles(query):
  if Chem.MolFromSmiles(query) is not None:
    return query
  try:
    encoded = urllib.parse.quote(query.strip())
    p1 = "https://nih.gov"
    p2 = f"/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
    with urllib.request.urlopen(p1 + p2, timeout=5) as response:
      return response.read().decode("utf-8").strip()
  except Exception:
    return None


def extract_upgraded_features(mol):
  return {
      "MolWt": Descriptors.MolWt(mol),
      "LogP": Descriptors.MolLogP(mol),
      "TPSA": Descriptors.TPSA(mol),
      "HDonors": Descriptors.NumHDonors(mol),
      "HAcceptors": Descriptors.NumHAcceptors(mol),
      "RotBonds": Descriptors.NumRotatableBonds(mol),
      "AromaticRings": Descriptors.NumAromaticRings(mol),
      "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
      "FractionCSP3": Descriptors.FractionCSP3(mol),
      "ValenceElectrons": Descriptors.NumValenceElectrons(mol),
  }


presets = {
    "Custom Input": "",
    "Atrazine (FL Agricultural Herbicide)": "CCNc1nc(Cl)nc(NC(C)C)n1",
    "Caffeine (Wastewater / Aquifer Tracer)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Aspirin (Pharmaceutical)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "DDT (Legacy Persistent Pesticide)": "Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl",
    "Benzo[a]pyrene (Combustion Pollutant)": "c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45",
}

choice = st.selectbox(
    "Select a preset chemical or enter your own:", list(presets.keys())
)
user_input = st.text_input(
    "Enter a Chemical Name (e.g., Ibuprofen, Glucose) OR SMILES String:",
    value=presets[choice] if choice != "Custom Input" else "Ibuprofen",
)

if user_input:
  smiles_input = name_to_smiles(user_input)
  if smiles_input is None:
    st.error(
        "Could not find that chemical name or SMILES string. Check spelling and"
        " try again!"
    )
  else:
    mol = Chem.MolFromSmiles(smiles_input)
    st.caption(f"Resolved SMILES: `{smiles_input}`")

    feats = pd.DataFrame([extract_upgraded_features(mol)])
    pred_logS = model.predict(feats).item()
    logP = feats["LogP"].item()
    arom_rings = feats["AromaticRings"].item()

    col1, col2 = st.columns(2)
    with col1:
      st.image(
          Draw.MolToImage(mol, size=(300, 300)),
          caption="2D Molecular Structure",
      )
    with col2:
      st.metric("Predicted Water Solubility (logS)", f"{pred_logS:.2f} mol/L")
      st.metric("Lipophilicity / Polarity (LogP)", f"{logP:.2f}")
      st.metric("Molar Mass", f"{feats['MolWt'].item():.2f} g/mol")

    st.subheader("Environmental & Florida Aquifer Fate")
    risk_found = False
    if pred_logS > -4.0 and logP < 2.5:
      st.warning(
          "⚠️ **HIGH GROUNDWATER LEACHING RISK:** Mobile in sandy Florida"
          " soils/aquifers."
      )
      risk_found = True
    if logP >= 3.5 and pred_logS <= -3.5:
      st.error(
          "🚨 **HIGH BIOACCUMULATION RISK:** Lipophilic; persists in aquatic"
          " food webs."
      )
      risk_found = True
    if arom_rings >= 3:
      st.warning(
          "⚠️ **PERSISTENT POLYCYCLIC AROMATIC:** Resists rapid microbial"
          " biodegradation."
      )
      risk_found = True
    if not risk_found:
      st.success("✅ **MODERATE / LOW ENVIRONMENTAL PERSISTENCE**")
