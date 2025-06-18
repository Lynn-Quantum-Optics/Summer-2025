# Overview
This folder contains a legacy data processing files from before Spring/Summer 2025.

# Important Files
- legacy_process_expt.py: this file can be used to process data files with the naming format from before Summer 2025, files containing smaller amounts of data, and data sets for states of the form phi = np.cos(eta)*PHI_PLUS + np.exp(1j*chi)*np.sin(eta)*PHI_MINUS.
- process_expt_with_legacy_rho_methods.py: this file can be used to process data using the naming & file conventions from Summer 2025 but with the legacy version of the rho methods file.
- full_tomo.py, rho_methods.py, sample_rho.py: legacy files imported in the data processing files