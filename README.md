# HSBC-UCL

This repository contains the files of my UCL MSc Computational Finance summer project in collaboration with the eFX quant team at HSBC. This project builds on the open source mbt_gym library available at: https://github.com/JJJerome/mbt_gym.

My contributions include:
1. A short-term-alpha midprice model (class name: AlphaAdverseSelecetionMidpriceModel) available in the midprice_models.py file in the stochastic_processes folder.
2. A stochastic exponential fill probability model (class name: StochasticExponentialFillFunction) available in the fill_probability_models_py file in the stochastic_processes folder.
3. Notebooks that include testing the new models, creating market making agents using different actor-critic algorithms from Stable Baselines 3, and performing hyperparameter optimisation.
