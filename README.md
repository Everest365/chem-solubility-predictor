# 🧪 AI Chemical Solubility & Florida Aquifer Risk Screener

An interactive machine-learning web application that predicts aqueous solubility ($\log S$) from molecular SMILES strings and evaluates environmental fate—specifically **groundwater leaching risk in sandy Florida aquifers**, **bioaccumulation potential**, and **polycyclic aromatic persistence**.

🔗 **Live Web App:** [Launch Interactive Screener](https://streamlit.app)

## 🔬 Scientific Methodology & Model Performance
Trained on the **Delaney ESOL dataset (1,128 real chemical compounds)** using **RDKit** and **Scikit-Learn**:
- **Baseline Model (6 physicochemical descriptors):** $R^2 = 0.853$ | $\text{RMSE} = 0.832\ \log\text{mol/L}$
- **Upgraded Model (10 physicochemical descriptors):** $R^2 = 0.862$ | $\text{RMSE} = 0.807\ \log\text{mol/L}$

Adding structural descriptors for **planar aromatic ring stacking (`NumAromaticRings`)**, **3D tetrahedral geometry (`FractionCSP3`)**, **molecular bulk (`HeavyAtomCount`)**, and **valence electron count (`NumValenceElectrons`)** measurably improved predictive accuracy and reduced prediction error.

## 🛠️ Tech Stack
- **Cheminformatics:** RDKit
- **Machine Learning:** Scikit-Learn (Random Forest Regressor), Pandas, Joblib
- **Web Deployment:** Streamlit Community Cloud
