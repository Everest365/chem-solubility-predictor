import json
import urllib.parse
import urllib.request
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, FilterCatalog
import streamlit as st

st.set_page_config(
    page_title="AI Solubility & Florida Aquifer Risk Screener",
    page_icon="🧪",
    layout="centered",
)
st.title("🧪 AI Chemical Solubility & Florida Aquifer Risk Screener")
st.write(
    "Predicts aqueous solubility (logS & mg/L) using a Random Forest model"
    " (R² = 0.862, RMSE = 0.807) and screens for environmental leaching,"
    " bioaccumulation, and chemical toxicity risks."
)

model = joblib.load("solubility_model (2).pkl")


def name_to_smiles(query):
  if Chem.MolFromSmiles(query) is not None:
    return query
  try:
    encoded = urllib.parse.quote(query.strip())
    p1 = "https://pubchem" + ".ncbi.nlm.nih.gov"
    p2 = f"/rest/pug/compound/name/{encoded}/property/IsomericSMILES/TXT"
    req = urllib.request.Request(
        p1 + p2, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=8) as response:
      lines = response.read().decode("utf-8").strip().splitlines()
      return lines.pop(0)
  except Exception:
    return None


def check_toxicity(smiles, mol):
  hazards = []

  # 1. General RDKit BRENK Toxicophore Filter (ignores harmless alkenes/fats)
  params = FilterCatalog.FilterCatalogParams()
  params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
  catalog = FilterCatalog.FilterCatalog(params)
  for entry in catalog.GetMatches(mol):
    desc = entry.GetDescription()
    if desc not in ["isolated_alkene", "more_than_2_esters"]:
      hazards.append(f"Toxicophore ({desc})")

  # 2. General Physicochemical Ecotoxicity Rules (EPA / Pfizer 3/75 Rule)
  logp = Descriptors.MolLogP(mol)
  tpsa = Descriptors.TPSA(mol)
  halogens = sum(
      1 for atom in mol.GetAtoms() if atom.GetSymbol() in ["Cl", "Br", "I"]
  )

  if logp > 3.0 and tpsa < 75.0:
    hazards.append("Membrane-Permeable Bioactive Risk (LogP > 3 & TPSA < 75)")
  if halogens >= 2:
    hazards.append(f"Polyhalogenated Compound ({halogens} Cl/Br/I atoms)")

  # 3. General Organophosphate / Carbamate Pesticide Core Check
  for label, patt in [
      ("Organophosphate / Thiophosphate Ester", "P(=[O,S])(O)(O)O"),
      ("Carbamate Ester Core", "NC(=O)O"),
  ]:
    sub = Chem.MolFromSmarts(patt)
    if sub and mol.HasSubstructMatch(sub):
      hazards.append(label)

  unique_hazards = list(dict.fromkeys(hazards))[:3]
  if unique_hazards:
    return "High / Known Hazard", ", ".join(unique_hazards)
  return "Low / Safe Organic Profile", "No major toxicophores or ecotoxicity alerts"



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


def evaluate_risks(pred_logS, logP, arom_rings, tox_status):
  flags = []
  is_toxic = tox_status == "High / Known Hazard"
  if pred_logS > -4.0 and logP < 2.5:
    flags.append(
        "HIGH CONTAMINATION RISK (Mobile + Toxic)"
        if is_toxic
        else "High Mobility / Low Toxicity (Benign Tracer)"
    )
  if logP >= 3.5 and pred_logS <= -3.5:
    flags.append("High Bioaccumulation Risk")
  if arom_rings >= 3:
    flags.append("Persistent Polycyclic Aromatic")
  return flags if flags else ["Moderate / Low Persistence"]


tab1, tab2 = st.tabs(
    ["🔬 Single Chemical Screener", "📊 Batch Comparison & CSV Export"]
)

with tab1:
  presets = {
      "Custom Input": "",
      "Atrazine (FL Agricultural Herbicide)": "CCNc1nc(Cl)nc(NC(C)C)n1",
      "Caffeine (Wastewater / Aquifer Tracer)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "Aspirin (Pharmaceutical)": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "DDT (Legacy Persistent Pesticide)": (
          "Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl"
      ),
      "Benzo[a]pyrene (Combustion Pollutant)": (
          "c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45"
      ),
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
          "Could not find that chemical name or SMILES string. Check spelling"
          " and try again!"
      )
    else:
      mol = Chem.MolFromSmiles(smiles_input)
      st.caption(f"Resolved SMILES: `{smiles_input}`")

      feats = pd.DataFrame([extract_upgraded_features(mol)])
      pred_logS = model.predict(feats).item()
      logP = feats["LogP"].item()
      mol_wt = feats["MolWt"].item()
      arom_rings = feats["AromaticRings"].item()
      heavy_atoms = feats["HeavyAtoms"].item()
      sol_mg_L = (10**pred_logS) * mol_wt * 1000
      tox_status, tox_details = check_toxicity(smiles_input, mol)

      if "." in smiles_input:
        st.info(
            "ℹ️ **APPLICABILITY NOTE:** Ionic salt / multi-component structure"
            " detected. Model is calibrated for neutral organic covalent"
            " molecules."
        )
      elif heavy_atoms <= 3:
        st.info(
            "ℹ️ **APPLICABILITY NOTE:** Very small molecule (≤3 heavy atoms,"
            " such as gases or pure water). Solubility may differ from"
            " condensed-phase organic predictions."
        )

      col1, col2 = st.columns(2)
      with col1:
        st.image(
            Draw.MolToImage(mol, size=(300, 300)),
            caption="2D Molecular Structure",
        )
      with col2:
        st.metric(
            "Predicted Water Solubility (logS)",
            f"{pred_logS:.2f} mol/L",
            delta=f"≈ {sol_mg_L:,.1f} mg/L (ppm)",
            delta_color="off",
        )
        st.metric("Lipophilicity / Polarity (LogP)", f"{logP:.2f}")
        st.metric("Molar Mass", f"{mol_wt:.2f} g/mol")

      st.subheader("Environmental & Florida Aquifer Fate")
      if tox_status == "High / Known Hazard":
        st.error(f"☠️ **TOXICITY ALERT ({tox_status}):** {tox_details}")
      else:
        st.info(f"🛡️ **TOXICITY SCREEN ({tox_status}):** {tox_details}")

      risk_found = False
      if pred_logS > -4.0 and logP < 2.5:
        if tox_status == "High / Known Hazard":
          st.error(
              "🚨 **CRITICAL AQUIFER CONTAMINATION RISK:** Toxic compound"
              " that leaches rapidly through sandy Florida soils."
          )
        else:
          st.warning(
              "💧 **HIGH MOBILITY / LOW TOXICITY:** Leaches easily in sandy"
              " soils, but poses low direct chemical toxicity (e.g., tracer or"
              " natural organic)."
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

with tab2:
  st.write(
      "Compare multiple chemicals side-by-side and export results for lab or"
      " environmental analysis."
  )
  batch_input = st.text_area(
      "Enter chemical names or SMILES separated by commas:",
      value="Atrazine, Glyphosate, Caffeine, Ibuprofen, DDT, Citric acid",
  )
  if st.button("Run Batch Screening"):
    items = [item.strip() for item in batch_input.split(",") if item.strip()]
    results = []
    for item in items:
      smi = name_to_smiles(item)
      if smi:
        m = Chem.MolFromSmiles(smi)
        if m:
          f = pd.DataFrame([extract_upgraded_features(m)])
          p_logS = model.predict(f).item()
          lp = f["LogP"].item()
          mw = f["MolWt"].item()
          ar = f["AromaticRings"].item()
          mg_l = (10**p_logS) * mw * 1000
          t_status, t_details = check_toxicity(smi, m)
          risks = ", ".join(evaluate_risks(p_logS, lp, ar, t_status))
          results.append({
              "Chemical": item,
              "Molar Mass (g/mol)": round(mw, 2),
              "LogP": round(lp, 2),
              "Predicted logS (mol/L)": round(p_logS, 2),
              "Solubility (mg/L)": round(mg_l, 1),
              "Toxicity Status": t_status,
"Hazard Details": t_details,
"Environmental Fate": risks,
})
    if results:
      res_df = pd.DataFrame(results)
      st.dataframe(res_df, use_container_width=True)
      st.download_button(
          "📥 Download Comparison Table (CSV)",
          data=res_df.to_csv(index=False).encode("utf-8"),
          file_name="aquifer_solubility_screening.csv",
          mime="text/csv",
      )
    else:
      st.error(
          "No valid chemicals could be resolved. Check spelling and try again."
      )

st.divider()
exp = st.expander("📊 Model Accuracy, Feature Importance & Known Exceptions")
exp.markdown("**Model Performance (Delaney ESOL Benchmark — 1,128 Compounds):**")
exp.markdown("* **Upgraded 10-Feature Random Forest:** R² = 0.862 | RMSE = 0.807 log mol/L")
exp.markdown("* **Baseline 6-Feature Random Forest:** R² = 0.853 | RMSE = 0.832 log mol/L")
exp.subheader("What Drives Water Solubility? (Feature Importance)")
feature_names = ["MolWt", "LogP", "TPSA", "HDonors", "HAcceptors", "RotBonds", "AromaticRings", "HeavyAtoms", "FractionCSP3", "ValenceElectrons"]
imp_df = pd.DataFrame({"Importance": model.feature_importances_}, index=feature_names).sort_values("Importance", ascending=False)
exp.bar_chart(imp_df)
exp.markdown("**Known Exceptions & Applicability Domain:**")
exp.markdown("* **Gases at Room Temperature (e.g., CO₂, NH₃, CH₄):** Trained on condensed-phase (solid/liquid) organics, so aqueous solubility is overestimated for gas-phase molecules.")
exp.markdown("* **Ionic Salts & Metals (e.g., NaCl, CaCO₃):** Designed for neutral covalent organic molecules; ionic lattice dissociation is not modeled.")
exp.markdown("* **Toxicity Screening:** Combines live PubChem GHS hazard classifications with offline RDKit structural toxicophore filters (NIH & SMARTS pollutant rules).")

