# Deep learning-based Causal Inference in Non-linear Multivariate Time series 

This repository contains code for our paper accepted in ICMLA 2021: Causal Inference in Non-linear Time-series using Deep Networks and Knockoff Counterfactuals by Wasim Ahmad, Maha Shadaydeh and Joachim Denzler.

- The work can be cited using below citation information.

```
@inproceedings{ahmad2021causal,
  title={Causal inference in non-linear time-series using deep networks and knockoff counterfactuals},
  author={Ahmad, Wasim and Shadaydeh, Maha and Denzler, Joachim},
  booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={449--454},
  year={2021},
  organization={IEEE}
}
```


## Overview

We discover full causal graph in multivariate nonlinear systems by testing model invariance against Knockoffs-based interventional environments:
1. First we train deep network <img src="https://render.githubusercontent.com/render/math?math=f_i"> using data from observational environment <img src="https://render.githubusercontent.com/render/math?math=E_i">.
2. Then we expose the model to Knockoffs-based interventional environments <img src="https://render.githubusercontent.com/render/math?math=E_k">. 
3. For each pair variables {<img src="https://render.githubusercontent.com/render/math?math=z_i">, <img src="https://render.githubusercontent.com/render/math?math=z_j">} in nonlinear system, we test model invariance across environments. 
4. We perform KS test over distribution <img src="https://render.githubusercontent.com/render/math?math=R_i">, <img src="https://render.githubusercontent.com/render/math?math=R_k"> of model residuals in various environments. 
Our NULL hypothesis is that variable <img src="https://render.githubusercontent.com/render/math?math=z_i"> does not cause <img src="https://render.githubusercontent.com/render/math?math=z_j">, 
<img src="https://render.githubusercontent.com/render/math?math=H_0">: <img src="https://render.githubusercontent.com/render/math?math=R_i"> = <img src="https://render.githubusercontent.com/render/math?math=R_k">, 
else the alternate hypothesis <img src="https://render.githubusercontent.com/render/math?math=H_1">: <img src="https://render.githubusercontent.com/render/math?math=R_i"> != <img src="https://render.githubusercontent.com/render/math?math=R_k">  is accepted.

<p align="center">
<img src="res/causality_demo.png" width=100% />
</p>

## Data
We test our method on synthetic as well as real data which can be found under `datasets/` directory. The synthetic data is generated using file `src/synthetic_dataset.py`. 
The real data we used is average daily discharges of rivers in the upper Danube basin, measurements of which are made available by the Bavarian Environmental Agency at
https://www.gkd.bayern.de.


## Code
`src/main.py` is our main file, where we model multivariate non-linear data using deep networks.
- `src/deepcause.py` for actual and counterfactual outcome generation using interventions.
- `src/knockoffs.py` generate knockoffs of the original variables.
- `src/daignostics.py` to determine the goodness of the generated knockoff copies.
- `DeepKnockoffs/` contains the knockoffs generation methods.
- `datasets/` contains the generated synthetic data and real dataset.
- `model/` contains trained models that we used for different datasets.


## Dependencies
`requirements.txt` contains all the packages that are related to the project.
To install them, simply create a new [conda](https://docs.conda.io/en/latest/) environment and type
```
pip install -r requirements.txt
```


## Acknowledgement

This work is funded by the Carl Zeiss Foundation within the scope of the program line "Breakthroughs: Exploring Intelligent Systems" for "Digitization � explore the basics, use applications" and the DFG grant SH 1682/1-1.
