# ECON570-Causal-Inference-WFH
Causal inference of remote work on vehicle ownership using Double Lasso and Causal Forest.

# ECO 570: Causal Inference on Remote Work and Vehicle Ownership

##  Project Overview
This repository contains the data processing and econometric modeling code for the ECO 570 (Big Data Economics) final project at USC. The study investigates the causal effect of Working From Home (WFH) on household vehicle ownership in Los Angeles County.

## 🛠️ Methodology & Tools
To address high-dimensional confounding variables and identify heterogeneous treatment effects, we applied advanced machine learning techniques in causal inference:
* **Data Pipeline:** Python (`pandas`, `numpy`)
* **Econometric Models:** R (`hdm`, `grf`, `fixest`)
    * **Double Lasso (DML):** High-dimensional variable selection to mitigate omitted variable bias.
    * **Causal Forest (GRF):** Estimation of Conditional Average Treatment Effects (CATE) across different income and occupational groups.
    * **TWFE:** Two-Way Fixed Effects for baseline comparisons.

##  File Description
* `data_clean.py`: Master data cleaning pipeline. Merges 2019 ACS PUMS micro-data with 2021 LODES macro features, and engineers interaction terms.
* `modeling.R`: Contains baseline OLS, Propensity Score trimming, Double Lasso implementation, and Causal Forest modeling.

##  Key Findings
The empirical results suggest that remote work significantly reduces the demand for vehicle ownership among high-income households, whereas the effect is limited for low-income households.
