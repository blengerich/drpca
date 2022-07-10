# Differential Robust PCA

Motivation: Dimensionality reduction is an important task in bioinformatics studies. Common unsupervised methods like principal components analysis (PCA) extract axes of variation that are high-variance but do not necessarily differentiate experimental conditions. Methods of supervised discriminant analysis such as partial least squares (PLS-DA) effectively separate conditions, but are hamstrung by inflexibility and overfit to sample labels. We would like a simple method which repurposes the rich literature of component estimation for supervised dimensionality reduction.

![Toy Example](https://github.com/blengerich/drpca/blob/master/fig/toy.png?raw-true "Toy Example")

Results: We propose to address this problem by estimating principal components from a set of difference vectors rather than from the samples. Our method directly utilizes the PCA algorithm as a module, so we can incorporate any PCA variant for improved components estimation. Specifically, Robust PCA, which ameliorates the deleterious effects of noisy samples, improves recovery of components in this framework. We name the resulting method Differential Robust PCA (drPCA). We apply drPCA to several cancer gene expression datasets and find that it more accurately summarizes oncogenic processes than do standard methods such as PCA and PLS-DA.

![Differential PCA Selects Oncogenes](https://github.com/blengerich/drpca/blob/master/fig/oncogene.png?raw=true "Oncogene Selection")

This repository contains code for the experiments described in the paper. Each dataset has a main Jupyter notebook titled `test_$cancer.ipynb` which generates the results stored in the corresponding subdirectory.

More information is available in the [manuscript](https://www.biorxiv.org/content/10.1101/545798v1.abstract): 
```
@article{lengerich2019differential,
  title={Differential Principal Components Reveal Patterns of Differentiation in Case/Control Studies},
  author={Lengerich, Benjamin J and Xing, Eric P},
  journal={bioRxiv},
  pages={545798},
  year={2019},
  publisher={Cold Spring Harbor Laboratory}
}
```
