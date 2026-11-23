import json
import re
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
    "Predicts aqueous solubility (logS & mg/L) using a Random Forest model "
    "(R² = 0.862, RMSE = 0.807) and screens for environmental leaching, "
    "bioaccumulation, and chemical toxicity risks."
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
    warnings = []

    # Detect benign polyols / chlorinated sugars (e.g., Sucralose, Glucose)
    oh_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))
    ether_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OD2]([#6])[#6]")))
    is_polyol_sugar = (oh_count >= 4 and ether_count >= 1)

    # 1. Filtered RDKit BRENK Toxicophore Filter (excludes noisy pharma-only alerts)
    brenk_ignore = {
        "isolated_alkene",
        "more_than_2_esters",
        "phenol_ester",
        "aldehyde",
        "hydroquinone",
        "Aliphatic_long_chain",
        "Polycyclic_aromatic_hydrocarbon_1",
        "Polycyclic_aromatic_hydrocarbon_2",
        "Polycyclic_aromatic_hydrocarbon_3",
    }
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)
    for entry in catalog.GetMatches(mol):
        desc = entry.GetDescription()
        if desc in brenk_ignore:
            continue
        if desc == "alkyl_halide" and is_polyol_sugar:
            continue
        hazards.append(f"Toxicophore ({desc})")

    # 2. Environmental Pollutant & Pesticide SMARTS Rules
    env_smarts = [
        ("Organophosphate / Thiophosphate Ester", "P(=[O,S])(O)(O)O"),
        ("Carbamate Ester Core", "NC(=O)O"),
        ("Chlorotriazine Herbicide (Atrazine/Simazine)", "c1nc(Cl)nc(N)n1"),
        ("Chlorophenoxy Herbicide (2,4-D class)", "OC(=O)COc1ccc(Cl)cc1"),
        ("Benzene / BTEX Aromatic Hydrocarbon", "[cR1]1[cR1][cR1][cR1][cR1][cR1]1"),
        ("True All-Carbon PAH (2+ Fused Rings)", "c1ccc2ccccc2c1"),
        ("Phthalate Plasticizer Ester", "O=C(O[#6])c1ccccc1C(=O)O[#6]"),
        ("Benzophenone UV Filter (Oxybenzone class)", "O=C(c1ccccc1)c2ccccc2"),
        ("Bicyclic Alkaloid Neurotoxin", "[NX3;R2;!a]"),
    ]

    for label, patt in env_smarts:
        sub = Chem.MolFromSmarts(patt)
        if sub and mol.HasSubstructMatch(sub):
            # Only flag bare BTEX if the molecule consists strictly of C and H
            if label == "Benzene / BTEX Aromatic Hydrocarbon":
                if any(a.GetSymbol() not in ["C", "H"] for a in mol.GetAtoms()):
                    continue
            hazards.append(label)

    # 3. Physicochemical & Halogen Rules
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    halogens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ["Cl", "Br", "I"])

    if halogens >= 2 and not is_polyol_sugar:
        hazards.append(f"Polyhalogenated Compound ({halogens} Cl/Br/I atoms)")

    if logp > 3.0 and tpsa < 75.0:
        warnings.append("Membrane-Permeable / Lipophilic (LogP > 3 & TPSA < 75)")

    unique_hazards = list(dict.fromkeys(hazards))[:3]
    unique_warnings = list(dict.fromkeys(warnings))

    if unique_hazards:
        return "High / Known Hazard", ", ".join(unique_hazards)
    if unique_warnings:
        return "Moderate / Bioactive Lipophilic", ", ".join(unique_warnings)
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


def evaluate_risks(pred_logS, logP, arom_rings, tox_status, tox_details="", mol=None):
    flags = []
    is_toxic = tox_status == "High / Known Hazard"

    is_pfas = "Perfluorinated_chain" in tox_details
    is_polyhalogenated = "Polyhalogenated Compound" in tox_details
    is_cation = mol is not None and any(a.GetFormalCharge() > 0 for a in mol.GetAtoms())

    # 1. Soil Binding vs. Groundwater Mobility
    if is_cation:
        flags.append("LOW LEACHING (Cation Binds Tightly to Soil Clay)")
    elif is_pfas:
        flags.append("HIGH AQUIFER MOBILITY + PERSISTENCE (Ionized PFAS)")
    elif pred_logS > -4.0 and logP < 2.5:
        if is_toxic:
            flags.append("HIGH CONTAMINATION RISK (Mobile + Toxic)")
        else:
            flags.append("High Mobility / Low Toxicity (Benign Tracer)")

    # 2. Bioaccumulation
    if (logP >= 3.5 and pred_logS <= -3.5) or is_pfas:
        flags.append("High Bioaccumulation Risk")

    # 3. Persistence (True PAHs or Chlorinated Solvents)
    if "True All-Carbon PAH" in tox_details:
        flags.append("Persistent Polycyclic Aromatic")
    elif is_polyhalogenated and not is_pfas:
        flags.append("High Groundwater Persistence (Recalcitrant Organohalide)")

    return flags if flags else ["Moderate / Low Persistence"]


tab1, tab2 = st.tabs(
    ["🔬 Single Chemical Screener", "📊 Batch Comparison & CSV Export"]
)

with tab1:
    presets = {
        "Custom Input": "",
        "Atrazine (FL Agricultural Herbicide)": "CCNc1nc(Cl)nc(NC(C)C)n1",
        "Caffeine (Wastewater / Aquifer Tracer)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Sucralose (Benign Wastewater Tracer)": "OC[C@H]1O[C@H](O[C@]2(CCl)O[C@H](CCl)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1Cl",
        "PFOS (Persistent PFAS Forever Chemical)": "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "Aspirin (Pharmaceutical)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "DDT (Legacy Persistent Pesticide)": "Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl",
        "Benzo[a]pyrene (Combustion Pollutant)": "c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45",
    }

    choice = st.selectbox(
        "Select a preset chemical or enter your own:", list(presets.keys())
    )
    user_input = st.text_input(
        "Enter a Chemical Name (e.g., Ibuprofen, Glucose) OR SMILES String:",
        value=presets[choice] if choice != "Custom Input" else "",
    )

    if user_input:
        smiles_input = name_to_smiles(user_input)
        if smiles_input is None:
            st.error(
                "Could not find that chemical name or SMILES string. Check spelling "
                "and try again!"
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
                    "ℹ️ **APPLICABILITY NOTE:** Ionic salt / multi-component structure "
                    "detected. Model is calibrated for neutral organic covalent "
                    "molecules."
                )
            elif heavy_atoms <= 3:
                st.info(
                    "ℹ️ **APPLICABILITY NOTE:** Very small molecule (≤3 heavy atoms, "
                    "such as gases or pure water). Solubility may differ from "
                    "condensed-phase organic predictions."
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
            elif tox_status == "Moderate / Bioactive Lipophilic":
                st.warning(f"⚠️ **TOXICITY SCREEN ({tox_status}):** {tox_details}")
            else:
                st.info(f"✅ **TOXICITY SCREEN ({tox_status}):** {tox_details}")

            fate_flags = evaluate_risks(
                pred_logS, logP, arom_rings, tox_status, tox_details, mol
            )
            for flag in fate_flags:
                if "HIGH" in flag or "Persistent" in flag or "Bioaccumulation" in flag:
                    st.error(f"🚨 **{flag}**")
                elif "LOW LEACHING" in flag or "Mobility" in flag:
                    st.warning(f"💧 **{flag}**")
                else:
                    st.success(f"🌿 **{flag}**")

with tab2:
    st.write(
        "Compare multiple chemicals side-by-side and export results for lab or "
        "environmental analysis."
    )
    batch_input = st.text_area(
        "Enter chemical names or SMILES separated by commas or new lines:",
        value="Atrazine, Sucralose, Caffeine, PFOS, Benzene, Riboflavin, Paraquat, DDT",
    )
    if st.button("Run Batch Screening"):
        # Splits on newlines or commas followed by whitespace/letters (preserves names like 2,4-D or 1,4-Dioxane)
        raw_items = re.split(r"[\n;]+|,(?!\d)", batch_input)
        items = [item.strip() for item in raw_items if item.strip()]
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
                    risks = ", ".join(
                        evaluate_risks(p_logS, lp, ar, t_status, t_details, m)
                    )
                    results.append(
                        {
                            "Chemical": item,
                            "Molar Mass (g/mol)": round(mw, 2),
                            "LogP": round(lp, 2),
                            "Predicted logS (mol/L)": round(p_logS, 2),
                            "Solubility (mg/L)": round(mg_l, 1),
                            "Toxicity Status": t_status,
                            "Hazard Details": t_details,
                            "Environmental Fate": risks,
                        }
                    )
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
            st.error("No valid chemicals could be resolved. Check spelling and try again.")

st.divider()
exp = st.expander("📊 Model Accuracy, Feature Importance & Known Exceptions")
exp.markdown("Model Performance (Delaney ESOL Benchmark — 1,128 Compounds):")
exp.markdown("Upgraded 10-Feature Random Forest: R² = 0.862 | RMSE = 0.807 log mol/L")
exp.markdown("Baseline 6-Feature Random Forest: R² = 0.853 | RMSE = 0.832 log mol/L")
exp.subheader("What Drives Water Solubility? (Feature Importance)")
feature_names = [
"MolWt",
"LogP",
"TPSA",
"HDonors",
"HAcceptors",
"RotBonds",
"AromaticRings",
"HeavyAtoms",
"FractionCSP3",
"ValenceElectrons",
]
imp_df = pd.DataFrame(
{"Importance": model.feature_importances_}, index=feature_names
).sort_values("Importance", ascending=False)
exp.bar_chart(imp_df)
exp.markdown("Known Exceptions & Applicability Domain:")
exp.markdown(
"Gases at Room Temperature (e.g., CO₂, NH₃, CH₄): Trained on condensed-phase "
"(solid/liquid) organics, so aqueous solubility is overestimated for gas-phase molecules."
)
exp.markdown(
"Ionic Salts & Metals (e.g., NaCl, CaCO₃): Designed for neutral covalent organic "
"molecules; ionic lattice dissociation is not modeled."
)
exp.markdown(
"Toxicity & Fate Screening: Combines filtered RDKit structural alerts with "
"environmental pollutant SMARTS rules (PFAS, triazines, BTEX, PAHs) and soil-leaching overrides."
)
