# Transfer_Payments_HANK
Implementation of Transfer Payment shocks into the IKC HANK framework with means testing and Austrian calibration, trying to emulate the "Klimabonus 2021"


This is the Implementation file of "Investigating the impact of income transfers on the
Austrian economy through a HANK-DSGE framework" (2026) and based on:

Taken from the repo of the original work:
"This repository replicates the figures and tables in "[The Intertemporal Keynesian Cross](https://shade-econ.github.io/ikc/ikc.pdf)" (Auclert, Rognlie, and Straub JPE 2024). \[[Published version](https://doi.org/10.1086/732531)\]

The code requires the [Sequence Space Jacobian (SSJ) toolkit](https://github.com/shade-econ/sequence-jacobian/) version 1.0, in addition to standard numerical Python packages (`numpy`, `scipy`, `matplotlib`, `numba`, `pandas`) and Jupyter notebooks. 
We have tested it using Python 3.10+. The SSJ toolkit can be installed using `pip install sequence-jacobian`; please see the toolkit site for additional instructions."


102 Replication file is the file necessary to compute IRFs, the DAG and cumulative multipliers of section 7 and the Appendix of the thesis. 
To change the calibrations of A/Z or other variables/parameters, like the income process, refer to calibration.py. 
models_heterogenous.py provides the possibility of changing the eligibility criterion via designating the threshold m of households via the parameter N_Test
