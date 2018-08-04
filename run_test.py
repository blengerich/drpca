import numpy as np
from contrastive import CPCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, FastICA
from sklearn.metrics import silhouette_score
from scipy.linalg import svd
import pandas as pd
import time
import os
from pyensembl import ensembl_grch38
from rpca import R_pca
from supervisedPCA import supervised_pca
import scipy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize


plt.rcParams['savefig.dpi'] = 300
cancer_type = "Combined"
split_by = "Case_Control"
fname_base = "tissue_type"

test_tissue_type = True
test_case_control = False

plot_clusters = True
plot_selection = True
plot_onc_selection = True
write_enrichment = True
plot_singular_values = False

add_unmatched = True # only for GBM
pct_train = 0.6
split_in_order = False

def load_data(data_dir):
    fg_data = np.load("{}cases.npy".format(data_dir))
    bg_data = np.load("{}controls.npy".format(data_dir))
    matches = np.load("{}matches.npy".format(data_dir))
    try:
        stages  = np.load("{}cases_stages.npy".format(data_dir))
    except FileNotFoundError:
        stages = np.array([])
    if split_by == "Stage":
        bg_data = np.squeeze(fg_data[np.where(stages == "stage iv"), :])
        fg_data = np.squeeze(fg_data[np.where(stages == "stage i"), :])
        print(bg_data.shape)
        print(fg_data.shape)
        matches = []
        fname_base = "stage"
    return fg_data, bg_data, matches, stages

if cancer_type == "Combined":
    fg_data_brca, bg_data_brca, matches_brca, stages_brca = load_data("BRCA/")
    fg_data_luad, bg_data_luad, matches_luad, stages_luad = load_data("LUAD/")
    fg_data_gbm, bg_data_gbm, matches_gbm, stages_gbm = load_data("GBM/")
    foreground_data = np.vstack((fg_data_brca, fg_data_luad, fg_data_gbm))
    background_data = np.vstack((bg_data_brca, bg_data_luad, bg_data_gbm))
    matches_luad[:, 0] += len(bg_data_brca)
    matches_luad[:, 1] += len(fg_data_brca)
    if len(matches_gbm > 0):
        matches_gbm[:, 0] += len(bg_data_brca) + len(bg_data_luad)
        matches_gbm[:, 1] += len(fg_data_brca) + len(fg_data_luad)
        matches = np.vstack((matches_brca, matches_luad, matches_gbm))
    else:
        matches = np.vstack((matches_brca, matches_luad))
    cancer_types_fg = np.zeros((foreground_data.shape[0], 3))
    cancer_types_fg[:fg_data_brca.shape[0], 0] = 1
    cancer_types_fg[fg_data_brca.shape[0]:fg_data_brca.shape[0]+fg_data_luad.shape[0], 1] = 1
    cancer_types_fg[fg_data_brca.shape[0]+fg_data_luad.shape[0]:, 2] = 1

    #cancer_types_bg = np.zeros((background_data.shape[0], 3))
    #cancer_types_bg[:bg_data_brca.shape[0], 0] = 1
    #cancer_types_bg[bg_data_brca.shape[0]:bg_data_brca.shape[0]+bg_data_luad.shape[0], 1] = 1
    #cancer_types_bg[bg_data_brca.shape[0]+bg_data_luad.shape[0]:, 2] = 1
else:
    data_dir = "{}/".format(cancer_type)
    foreground_data, background_data, matches, stages = load_data(data_dir)


def to_one_hot_one_feature(U):
    """ Assumes U has a single feature.
    Returns matrix of size U.shape[0], number_unique + 1
    """
    as_set = set(U)
    print(as_set)
    set_as_list = list(as_set)
    one_hot = np.zeros((U.shape[0], len(as_set)))
    for i in range(U.shape[0]):
        one_hot[i, set_as_list.index(U[i])] = 1
    return one_hot


"""
pct_bad = 0.1
for i in range(len(matches)):
    if np.random.binomial(1, pct_bad):
        matches[i, 1] = (matches[i, 1]+1) % len(foreground_data)
"""
transcript_names = np.load("transcript_names.npy")
n_fg = len(foreground_data)
n_bg = len(background_data)

print(n_fg, n_bg, len(transcript_names))
transcript_names_converted = []
class Nil(object):
    pass
not_found = Nil()
not_found.gene_name = "Not Found"
for i, t_id in enumerate(transcript_names):
    try:
        transcript_names_converted.append(ensembl_grch38.gene_by_id(t_id[2:-3]))
    except ValueError:
        transcript_names_converted.append(not_found)
        foreground_data[:, i] = 0.
        background_data[:, i] = 0.

transcript_names = np.array(transcript_names_converted)
fg_mask = np.std(foreground_data, axis=0) > 1e-1
bg_mask = np.std(background_data, axis=0) > 1e-1
mask = np.logical_and(fg_mask, bg_mask)
name_mask = np.array([x != not_found for x in transcript_names])
mask = np.logical_and(mask, name_mask)
foreground_data = foreground_data[:, mask]
background_data = background_data[:, mask]
transcript_names = transcript_names[mask]
print(foreground_data.shape)
n_transcripts = 50000
foreground_data = foreground_data[:, :n_transcripts]
background_data = background_data[:, :n_transcripts]
transcript_names = transcript_names[:n_transcripts]
print(foreground_data.shape)
print(background_data.shape)

from sklearn.preprocessing import normalize
combined = np.vstack((foreground_data, background_data))
combined_normed = normalize(combined, axis=1)
combined_normed = normalize(combined_normed, axis=0)
del combined
foreground_data = combined_normed[:n_fg]
background_data = combined_normed[n_fg:]
"""
noise_level=0.0
if noise_level > 0:
    for i in range(np.min([n_fg, n_bg])):
        if np.random.binomial(1, noise_level):
            temp = background_data[i]
            background_data[i] = foreground_data[i]
            foreground_data[i] = temp
"""

if split_in_order:
    n_train_fg = int(pct_train*n_fg)
    n_train_bg = int(pct_train*n_bg)
    n_train_fg = len(fg_data_brca)
    n_train_bg = len(bg_data_brca)
    train_fg_idxs = np.array(range(n_train_fg))
    train_bg_idxs = np.array(range(n_train_bg))
    test_fg_idxs  = np.array(range(n_train_fg, n_fg))
    test_bg_idxs  = np.array(range(n_train_bg, n_bg))
    n_test_fg = len(test_fg_idxs)
    n_test_bg = len(test_bg_idxs)
else:
    train_fg_idxs, test_fg_idxs = train_test_split(list(range(n_fg)), test_size=1-pct_train)
    train_bg_idxs, test_bg_idxs = train_test_split(list(range(n_bg)), test_size=1-pct_train)
    n_train_fg = len(train_fg_idxs)
    n_test_fg  = len(test_fg_idxs)
    n_train_bg = len(train_bg_idxs)
    n_test_bg  = len(test_bg_idxs)


differential = np.zeros((1, foreground_data.shape[1]))
if len(matches) > 0:
    print("****Matched****")
    differential = np.array([foreground_data[i] - background_data[j] for [j, i] in matches if i in train_fg_idxs and j in train_bg_idxs])
    differential = np.vstack((differential, np.zeros_like(differential)))
if add_unmatched:
    n_differential = 1500
    print("****Add {} unmatched differences****".format(n_differential))
    unmatched = np.array([foreground_data[np.random.choice(train_fg_idxs)] - background_data[np.random.choice(train_bg_idxs)] for _ in range(n_differential)])
    unmatched = np.vstack((unmatched, np.zeros_like(unmatched)))
    differential = np.vstack((differential, unmatched))


differential = normalize(differential, axis=1)
differential = normalize(differential, axis=0)

max_n_components = 25
train_fg = foreground_data[train_fg_idxs]
train_bg = background_data[train_bg_idxs]
test_fg  = foreground_data[test_fg_idxs]
test_bg  = background_data[test_bg_idxs]
train_data = np.vstack((train_fg, train_bg))
test_data  = np.vstack((test_fg,  test_bg))
train_labels = np.ravel(np.vstack((np.ones((n_train_fg, 1)), np.zeros((n_train_bg, 1)))))
test_labels = np.ravel(np.vstack((np.ones((n_test_fg, 1)), np.zeros((n_test_bg, 1)))))

#train_cancer_types = np.vstack((cancer_types_fg[train_fg_idxs], cancer_types_bg[train_bg_idxs]))
#test_cancer_types  = np.vstack((cancer_types_fg[test_fg_idxs],  cancer_types_bg[test_bg_idxs]))
if cancer_type == "Combined":
    train_cancer_types = cancer_types_fg[train_fg_idxs]
    test_cancer_types  = cancer_types_fg[test_fg_idxs]

#good_stages_vals = np.logical_and(stages != "None", stages != 'not reported')
#good_stages_vals = np.logical_and(good_stages_vals, stages != 'stage x')
#good_stages_vals = np.logical_or(stages == "stage i", stages == "stage ii")
#good_stages_vals = np.logical_or(good_stages_vals, stages == "stage iii")
#good_stages_vals = stages == "stage i"
#good_stages_vals = np.logical_or(good_stages_vals, stages == "stage iv")
#foreground_data = foreground_data[good_stages_vals]
#stages = to_one_hot_one_feature(stages)

if not test_case_control and not test_tissue_type:
    stages_parsed = np.zeros((len(stages), 5))
    for i, stage in enumerate(stages):
        if "stage iv" in stage:
            stages_parsed[i, :] = np.array([0, 0, 0, 0, 1])
        elif "stage iii" in stage:
            stages_parsed[i, :] = np.array([0, 0, 0, 1, 0])
        elif "stage ii" in stage:
            stages_parsed[i, :] = np.array([0, 0, 1, 0, 0])
        elif "stage i" in stage:
            stages_parsed[i, :] = np.array([0, 1, 0, 0, 0])
        else:
            stages_parsed[i, :] = np.array([1, 0, 0, 0, 0])

    stages = stages_parsed
    train_stages = stages[train_fg_idxs]
    print(train_stages)
    test_stages  = stages[test_fg_idxs]
    train_stages = np.argmax(train_stages, axis=1)
    test_stages = np.argmax(test_stages, axis=1)
    print(train_stages)
    print(test_stages)


#all_data = np.vstack((train_data, test_data))
#train_data = np.vstack((foreground_data[:n_train_fg], background_data[:n_train_bg]))
#test_data  = np.vstack((foreground_data[n_train_fg:], background_data[n_train_bg:]))

def get_differential(data, numComponents=None):
        """Principal Components Analysis

        From: http://stackoverflow.com/a/13224592/834250

        Parameters
        ----------
        data : `numpy.ndarray`
            numpy array of data to analyse
        numComponents : `int`
            number of principal components to use

        Returns
        -------
        comps : `numpy.ndarray`
            Principal components
        evals : `numpy.ndarray`
            Eigenvalues
        evecs : `numpy.ndarray`
            Eigenvectors
        """
        m, n = data.shape
        data -= data.mean(axis=0)


        pca = PCA(n_components=numComponents)
        data_components = pca.fit_transform(data)
        return data_components, pca.singular_values_, pca.components_.T
        """

        print("Calculating cov")
        R = np.cov(data, rowvar=False)
        print("Finished cov")
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = np.linalg.eigh(R)
        print("Finished eigh.")
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        if numComponents is not None:
            evecs = evecs[:, :numComponents]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T, evals, evecs
        """

"""
pca = PCA(n_components=2)
#x = np.mean(foreground_data, axis=0) - np.mean(background_data, axis=0)
pca.fit(differential)
del differential
fg_diff_transformed = pca.transform(foreground_data)
good_indices = np.abs(fg_diff_transformed[:, 0]) < 10000
#good_indices = np.logical_and(good_indices, np.abs(fg_diff_transformed[:, 1]) < 80000)
mapping = []
for i in range(len(fg_diff_transformed)):
    if good_indices[i]:
        mapping.append(np.sum(good_indices[:i]))
    else:
        mapping.append(-1)
print("{} Good indices".format(np.sum(good_indices)))

foreground_data = foreground_data[good_indices]
differential = np.array([foreground_data[mapping[i]] - background_data[j] for [j, i] in matches])
differential = np.vstack((differential, np.array([background_data[j] - foreground_data[mapping[i]] for [j, i] in matches])))
"""


plt.suptitle(cancer_type, fontsize=28)
# Normal PCA
print("Fitting PCA...", end='')
t = time.time()
pca = PCA(n_components=max_n_components)
pca_train_reduced = pca.fit_transform(train_data)
print("Took {:.3f} seconds".format(time.time() - t))
pca_test_reduced = pca.transform(test_data)
pca_components = pca.components_
pca_sing = pca.singular_values_

print("Fitting ICA...", end='')
t = time.time()
ica = FastICA(n_components=max_n_components)
ica_train_reduced = ica.fit_transform(train_data)
print("Took {:.3f} seconds".format(time.time() - t))
ica_test_reduced  = ica.transform(test_data)
ica_components    = ica.components_.copy()

print("Fitting LDA...", end='')
t = time.time()
lda = LDA(n_components=max_n_components)
lda_train_reduced = lda.fit_transform(train_data, train_labels)
print("Took {:.3f} seconds".format(time.time() - t))
lda_test_reduced  = lda.transform(test_data)
lda_components    = np.tile(lda.scalings_.T, (max_n_components, 1))

print("Fitting SupPCA...", end='')
t = time.time()
sup_pca = supervised_pca.SupervisedPCAClassifier(n_components=max_n_components)
sup_pca.fit(train_data, train_labels)
sup_train_reduced = sup_pca.get_transformed_data(train_data)
sup_test_reduced  = sup_pca.get_transformed_data(test_data)
sup_components    = sup_pca.get_components()
print("Took {:.3f} seconds".format(time.time() - t))

print("Fitting PLS...", end='')
t = time.time()
plsr = PLSRegression(n_components=max_n_components, scale=True)
plsr.fit(train_data, train_labels)
plsr_train_reduced = plsr.x_scores_#.dot(plsr.x_weights_.T)
plsr_test_reduced  = plsr.transform(test_data)#.dot(plsr.x_weights_.T)
plsr_components    = plsr.x_weights_.T # n_components x n_features
print("Took {:.3f} seconds".format(time.time() - t))


print("Fitting CCA...", end='')
t = time.time()
from sklearn.cross_decomposition import CCA
cca = CCA(n_components=max_n_components, scale=True)
cca.fit(train_data, train_labels)
cca_components = cca.x_weights_.T
cca_train_reduced = train_data.dot(cca_components.T)#cca.predict(train_data)#.dot(cca_components)
cca_test_reduced  = test_data.dot(cca_components.T)#cca.predict(test_data)#.dot(cca_components)
print("Took {:.3f} seconds.".format(time.time() - t))

# Contrastive PCA
print("Fitting cPCA...", end='')
t = time.time()
mdl = CPCA(n_components=max_n_components)
cpca_preprocess = PCA(n_components=500)
train_data_preprocessed = cpca_preprocess.fit_transform(train_data)
test_data_preprocessed = cpca_preprocess.transform(test_data)
cpca_preprocess_components = cpca_preprocess.components_.copy()
cpca_train_fg_reduced, alpha = mdl.fit_transform(train_data_preprocessed[:n_train_fg], train_data_preprocessed[n_train_fg:],
    n_alphas=1, n_alphas_to_return=1, return_alphas=True)
alpha = alpha[0]
print("Took {:.3f} seconds".format(time.time() - t))
cpca_train_fg_reduced = cpca_train_fg_reduced[0]
cpca_train_bg_reduced = mdl.transform(train_data_preprocessed[n_train_fg:])[0]
cpca_train_reduced = np.vstack((cpca_train_fg_reduced, cpca_train_bg_reduced))
cpca_test_reduced = mdl.transform(test_data_preprocessed)[0]

pca = PCA(n_components=max_n_components)
pca.fit(mdl.fg_cov - alpha*mdl.bg_cov)
cpca_components = pca.components_.dot(cpca_preprocess_components)
#cpca_reduced = mdl.transform(all_data_preprocessed)[0]
#fg_cpca = cpca_reduced[:n_fg]
#bg_cpca = cpca_reduced[n_fg:]
#cpca_test_reduced = np.vstack((fg_cpca[n_train_fg:], bg_cpca[n_train_bg:]))

print("Fitting rPCA...", end='')
t = time.time()
rpca = R_pca(train_data, mu=1., lmbda=10.)
L, S = rpca.fit(max_iter=5000, iter_print=100)
_, rpca_evals, rpca_evecs = get_differential(L, max_n_components)
rpca_components = rpca_evecs.T
rpca_train_reduced = train_data.dot(rpca_evecs)
rpca_test_reduced  = test_data.dot(rpca_evecs)
print("Took {:.3f} seconds.".format(time.time() - t))

print("Fitting sPCA...", end='')
t = time.time()
#spca = SparsePCA(n_components=max_n_components, max_iter=10,
#    verbose=False, alpha=0.1, ridge_alpha=0.01)
#spca.fit(train_data)
print("Took {:.3f} seconds.".format(time.time() - t))
#spca_components = spca.components_
#spca_train_reduced = spca.transform(train_data)
#spca_test_reduced  = spca.transform(test_data)

print("Fitting dPCA...", end='')
t = time.time()
dpca = PCA(n_components=max_n_components)
dpca.fit(differential)
print("Took {:.3f} seconds.".format(time.time() - t))
dpca_train_reduced = dpca.transform(train_data)
dpca_test_reduced  = dpca.transform(test_data)
#fg_diff_transformed = dpca.transform(foreground_data)
#bg_diff_transformed = dpca.transform(background_data)
dpca_components = dpca.components_.copy()
diff_sing = dpca.singular_values_.copy()


print("Fitting dsPCA...", end='')
t = time.time()
#dspca = SparsePCA(n_components=max_n_components, max_iter=10,
#    verbose=False, alpha=0.1, ridge_alpha=0.01)
#dspca.fit(differential)
print("Took {:.3f} seconds.".format(time.time() - t))
#dspca_components = dspca.components_
#dspca_train_reduced = dspca.transform(train_data)
#dspca_test_reduced  = dspca.transform(test_data)


# drPCA
print("Fitting drPCA...", end='')
t = time.time()
rpca = R_pca(differential, mu=1., lmbda=10.)
L, S = rpca.fit(max_iter=5000, iter_print=100)
_, drpca_evals, drpca_evecs = get_differential(L, max_n_components)
drpca_components = drpca_evecs.T
drpca_train_reduced = train_data.dot(drpca_evecs)
drpca_test_reduced  = test_data.dot(drpca_evecs)
#fg_drpca = foreground_data.dot(drpca_evecs)
#bg_drpca = background_data.dot(drpca_evecs)
print("Took {:.3f} seconds.".format(time.time() - t))

"""
print("Fitting dICA...", end='')
t = time.time()
dica = FastICA(n_components=max_n_components, max_iter=1000)
dica.fit(differential)
print("Took {:.3f} seconds.".format(time.time() - t))
dica_components = dica.components_
dica_train_reduced = dica.transform(train_data)
dica_test_reduced  = dica.transform(test_data)
#dica_transformed = dica.transform(all_data)
"""

reduced = [(pca_train_reduced, pca_test_reduced, pca_components, "PCA"),
    (rpca_train_reduced, rpca_test_reduced, rpca_components, "rPCA"),
    #(spca_train_reduced, spca_test_reduced, spca_components, "sPCA"),
    (ica_train_reduced, ica_test_reduced, ica_components, "ICA"),
    (cpca_train_reduced, cpca_test_reduced, cpca_components,  "cPCA"),
    (sup_train_reduced, sup_test_reduced, sup_components, "Sup. PCA"),
    (cca_train_reduced, cca_test_reduced, cca_components, "CCA"),
    (plsr_train_reduced, plsr_test_reduced, plsr_components, "PLS-DA"),
    (lda_train_reduced, lda_test_reduced, lda_components, "LDA"),
    (dpca_train_reduced, dpca_test_reduced, dpca_components, "dPCA"),
    #(dspca_train_reduced, dspca_test_reduced, dspca_components, "dsPCA"),
    (drpca_train_reduced, drpca_test_reduced, drpca_components, "drPCA")]

for (_, _, comps, name) in reduced:
    print(name, comps.shape)

def plot_clusts(fg_train, bg_train, fg_test, bg_test, name):
    fig = plt.figure()
    if name == "LDA":
        plt.scatter(fg_train[:, 0], np.zeros_like(fg_train[:, 0]), marker='*', color='red', label='Train Case', alpha=0.3)
        plt.scatter(bg_train[:, 0], np.zeros_like(bg_train[:, 0]), marker='*', color='blue', label='Train Control', alpha=0.3)
        plt.scatter(fg_test[:, 0], np.zeros_like(fg_test[:, 0]), marker='+', color='orange', label='Test Case', alpha=0.3)
        plt.scatter(bg_test[:, 0], np.zeros_like(bg_test[:, 0]), marker='+', color='teal', label='Test Control', alpha=0.3)
    else:
        plt.scatter(fg_train[:, 0], fg_train[:, 1], marker='*', color='red', label='Train Case', alpha=0.3)
        plt.scatter(bg_train[:, 0], bg_train[:, 1], marker='*', color='blue', label='Train Control', alpha=0.3)
        plt.scatter(fg_test[:, 0], fg_test[:, 1], marker='+', color='orange', label='Test Case', alpha=0.3)
        plt.scatter(bg_test[:, 0], bg_test[:, 1], marker='+', color='teal', label='Test Control', alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/Clusters/{}_{}.png".format(cancer_type, fname_base, name))
    #plt.show()

if plot_clusters:
    helper = lambda train, test, name: plot_clusts(train[:n_train_fg, :2], train[n_train_fg:, :2],
                                            test[:n_fg-n_train_fg, :2], test[n_fg-n_train_fg:, :2], name)
    for (train, test, components, name) in reduced:
        helper(train, test, name)
    """
    plot_clusts(pca_train_reduced[:n_train_fg, :2], pca_train_reduced[n_train_fg:, :2],
                pca_test_reduced[:n_fg-n_train_fg, :2], pca_test_reduced[n_fg-n_train_fg:, :2], 'PCA')
    plot_clusts(rpca_train_reduced[:n_train_fg, :2], rpca_train_reduced[n_train_fg:, :2],
                rpca_test_reduced[:n_fg-n_train_fg, :2], rpca_test_reduced[n_fg-n_train_fg:, :2], 'rPCA')
    plot_clusts(spca_train_reduced[:n_train_fg, :2], spca_train_reduced[n_train_fg:, :2],
                spca_test_reduced[:n_fg-n_train_fg, :2], spca_test_reduced[n_fg-n_train_fg:, :2], 'sPCA')
    plot_clusts(ica_train_reduced[:n_train_fg, :2], ica_train_reduced[n_train_fg:, :2],
                ica_test_reduced[:n_fg-n_train_fg, :2], ica_test_reduced[n_fg-n_train_fg:, :2], 'ICA')
    plot_clusts(plsr_train_reduced[:n_train_fg, :2], plsr_train_reduced[n_train_fg:, :2],
                plsr_test_reduced[:n_fg-n_train_fg, :2], plsr_test_reduced[n_fg-n_train_fg:, :2], 'PLS')
    plot_clusts(lda_train_reduced[:n_train_fg, :2], lda_train_reduced[n_train_fg:, :2],
                lda_test_reduced[:n_fg-n_train_fg, :2], lda_test_reduced[n_fg-n_train_fg:, :2], 'LDA')
    plot_clusts(cpca_train_reduced[:n_train_fg, :2], cpca_train_reduced[n_train_fg:, :2],
                cpca_test_reduced[:n_fg-n_train_fg, :2], cpca_test_reduced[n_fg-n_train_fg:, :2], 'cPCA')
    plot_clusts(dpca_train_reduced[:n_train_fg, :2], dpca_train_reduced[n_train_fg:, :2],
                dpca_test_reduced[:n_fg-n_train_fg, :2], dpca_test_reduced[n_fg-n_train_fg:, :2], 'dPCA')
    plot_clusts(drpca_train_reduced[:n_train_fg, :2], drpca_train_reduced[n_train_fg:, :2],
                drpca_test_reduced[:n_fg-n_train_fg, :2], drpca_test_reduced[n_fg-n_train_fg:, :2], 'drPCA')
    plot_clusts(dspca_train_reduced[:n_train_fg, :2], dspca_train_reduced[n_train_fg:, :2],
                dspca_test_reduced[:n_fg-n_train_fg, :2], dspca_test_reduced[n_fg-n_train_fg:, :2], 'dsPCA')
    plot_clusts(dica_train_reduced[:n_train_fg, :2], dica_train_reduced[n_train_fg:, :2],
                dica_test_reduced[:n_fg-n_train_fg, :2], dica_test_reduced[n_fg-n_train_fg:, :2], 'dICA')
    """

# Plot the Rank Curve
cosmic_genes = set([])
with open("cosmic_gene_census.tsv", 'r') as gene_census:
    #for symbol, tissue in zip(df[["Gene Symbol"]].values, df[["Tumour Types(Somatic)"]].values):
    for line in gene_census:
        #if cancer_type == "BRCA":
        #    if "breast" in line.lower() or "brca" in line.lower(): #str(tissue[0]).lower():
        #        cosmic_genes.add(line.split('\t')[0].upper())
        cosmic_genes.add(line.split('\t')[0].upper())
#print(len(cosmic_genes))

def generate_curve(sorted_list):
    counts = []
    count = 0
    for (i, val) in sorted_list:
        if transcript_names[i].gene_name.upper() in cosmic_genes:
            #print(transcript_names[i].gene_name)
            count += 1
        counts.append(count)
    return counts

"""
counts_pca = generate_curve(sorted_pca)
counts_dpca = generate_curve(sorted_dpca)
counts_drpca = generate_curve(sorted_drpca)
counts_dspca = generate_curve(sorted_dspca)

plt.plot(counts_pca, label='PCA')
plt.plot(counts_dpca, label='dPCA')
plt.plot(counts_drpca, label='drPCA')
plt.title("{} Component 1".format(cancer_type), fontsize=28)
plt.xlabel("Variable Rank", fontsize=22)
plt.ylabel("Number of Oncogenes", fontsize=22)
plt.legend()
plt.show()
"""

"""
if plot_singular_values:
    fig = plt.figure()
    plt.plot(pca_sing, label="PCA")
    plt.plot(diff_sing, label="dPCA")
    plt.plot(drpca_evals, label="drPCA")
    plt.title("Singular Values")
    plt.show()
"""

def plot_list_top(n):
    sort_helper = lambda components: [i for i in sorted(enumerate(np.sum(np.abs(components[:n]), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]
    sorted_components = [sort_helper(components) for (train, test, components, name) in reduced]

    """
    weighted_components_pca = normal_components[:n]#*np.tile(np.expand_dims(np.square(pca_sing[:n]), 1), (1, combined.shape[1]))
    sorted_pca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_pca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_rpca = rpca_evecs.T[:n]#*np.tile(np.expand_dims(np.square(drpca_evals[:n]), 1), (1, combined.shape[1]))
    sorted_rpca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_rpca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_spca = spca_components[:n]#*np.tile(np.expand_dims(np.square(spca_sing[:n]), 1), (1, combined.shape[1]))
    sorted_spca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_spca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_ica = ica_components[:n]
    sorted_ica = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_ica), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_pls = plsr_components[:n]
    sorted_pls = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_pls), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    sorted_cpca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_cpca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_dpca = diff_components[:n]#*np.tile(np.expand_dims(np.square(diff_sing[:n]), 1), (1, combined.shape[1]))
    sorted_dpca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_dpca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_drpca = drpca_evecs.T[:n]#*np.tile(np.expand_dims(np.square(drpca_evals[:n]), 1), (1, combined.shape[1]))
    sorted_drpca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_drpca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_dspca = dspca_components[:n]#*np.tile(np.expand_dims(np.square(spca_sing[:n]), 1), (1, combined.shape[1]))
    sorted_dspca = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_dspca), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_lda = lda_components[:n] # TODO: this isn't a fair comparison because LDA only uses a single "component".
    sorted_lda = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_lda), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]

    weighted_components_dica = dica_components[:n]
    sorted_dica = [i[0] for i in sorted(enumerate(np.sum(np.abs(weighted_components_dica), axis=0)), key=lambda x: np.abs(x[1]), reverse=True)]
    """

    def write_to_file(sorted_components, name):
        os.makedirs("{}/Components/{}".format(cancer_type, name), exist_ok=True)
        with open("{}/Components/{}/{}_{}_top_{}.csv".format(cancer_type, name, fname_base, name, n), 'w') as csv_file:
            with open("{}/Components/{}/{}_{}_top_{}.tsv".format(cancer_type, name, fname_base, name, n), 'w') as tsv_file:
                for (idx, magnitude) in sorted_components:
                    print("{},{}".format(transcript_names[idx].gene_name, magnitude), file=csv_file)
                    print("{}\t{}".format(transcript_names[idx].gene_name, magnitude), file=tsv_file)


    if write_enrichment:
        for i, comps in enumerate(sorted_components):
            write_to_file(comps, reduced[i][-1])
        """
        write_to_file(weighted_components_pca, "PCA")
        write_to_file(weighted_components_rpca, "rPCA")
        write_to_file(weighted_components_spca, "sPCA")
        write_to_file(weighted_components_ica, "ICA")
        write_to_file(weighted_components_lda, "LDA")
        write_to_file(weighted_components_cpca, "cPCA")
        write_to_file(weighted_components_dpca, "dPCA")
        write_to_file(weighted_components_drpca, "drPCA")
        write_to_file(weighted_components_dspca, "dsPCA")
        write_to_file(weighted_components_dica, "dICA")
        write_to_file(weighted_components_pls, "PLS")
        """

    """
    counts_pca = generate_curve(sorted_pca)
    counts_rpca = generate_curve(sorted_rpca)
    counts_spca = generate_curve(sorted_spca)
    counts_cpca = generate_curve(sorted_cpca)
    counts_lda = generate_curve(sorted_lda)
    counts_dpca = generate_curve(sorted_dpca)
    counts_drpca = generate_curve(sorted_drpca)
    counts_dspca = generate_curve(sorted_dspca)
    counts_dica = generate_curve(sorted_dica)
    counts_ica  = generate_curve(sorted_ica)
    counts_pls  = generate_curve(sorted_pls)
    """


    if plot_selection:
        fig = plt.figure()
        #plt.title("{} Components 1-{}".format(cancer_type, n), fontsize=28)
        helper = lambda counts, name: plt.plot(counts, label=name, linestyle='dashed')
        for i, comps in enumerate(sorted_components):
            helper(np.array(generate_curve(comps)) + np.random.uniform(0, 1), reduced[i][-1])
        """
        plt.plot(np.array(counts_pca)-1., label='PCA', linestyle='dashed')
        plt.plot(counts_spca, label="sPCA", linestyle='dashed')
        plt.plot(np.array(counts_rpca)+1., label='rPCA', linestyle='dashed')
        plt.plot(counts_ica, label="ICA", linestyle='dashed')
        plt.plot(counts_lda, label='LDA', linestyle='dashed')
        plt.plot(counts_pls, label='PLS', linestyle='dashed')
        plt.plot(counts_cpca, label="cPCA", linestyle='dashed')
        plt.plot(counts_dpca, label='dPCA', linestyle='dashed')
        plt.plot(counts_dspca, label="dsPCA", linestyle='dashed')
        plt.plot(np.array(counts_drpca)+1., label='drPCA', linestyle='dashed')
        """
        #plt.plot(counts_dica, label="dICA", linestyle='dashed')
        plt.xlabel("Variable Rank", fontsize=22)
        plt.ylabel("Number of Oncogenes/TSGs", fontsize=22)
        plt.legend(fontsize=18)
        plt.savefig("{}/oncogene_selection_top_{}_{}".format(cancer_type, n, fname_base))
        #plt.show()

    return sorted_components

def plot_list_all(last):
    for n in range(last):
        sort_helper = lambda components: [i for i in sorted(enumerate(np.abs(components[n])), key=lambda x: np.abs(x[1]), reverse=True)]
        sorted_components = [sort_helper(components) for (train, test, components, name) in reduced]

        """
        weighted_components_pca = normal_components[n]
        sorted_pca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_pca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_rpca = rpca_evecs.T[n]#*np.tile(np.expand_dims(np.square(drpca_evals[:n]), 1), (1, combined.shape[1]))
        sorted_rpca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_rpca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_spca = spca_components[n]#*np.tile(np.expand_dims(np.square(spca_sing[:n]), 1), (1, combined.shape[1]))
        sorted_spca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_spca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_ica = ica_components[n]
        sorted_ica = [i[0] for i in sorted(enumerate(np.abs(weighted_components_ica)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_pls = plsr_components[n]
        sorted_pls = [i[0] for i in sorted(enumerate(np.abs(weighted_components_pls)), key=lambda x: np.abs(x[1]), reverse=True)]


        pca = PCA(n_components=n+1)
        pca.fit(mdl.fg_cov - alpha*mdl.bg_cov)
        weighted_components_cpca = pca.components_[n].dot(cpca_preprocess_components)
        sorted_cpca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_cpca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_dpca = diff_components[n]
        sorted_dpca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_dpca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_drpca = drpca_evecs.T[n]
        sorted_drpca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_drpca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_dspca = dspca_components[n]
        sorted_dspca = [i[0] for i in sorted(enumerate(np.abs(weighted_components_dspca)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_lda = lda_components[0] # Only the number of components as the number of classes?
        sorted_lda = [i[0] for i in sorted(enumerate(np.abs(weighted_components_lda)), key=lambda x: np.abs(x[1]), reverse=True)]

        weighted_components_dica = dica_components[n]
        sorted_dica = [i[0] for i in sorted(enumerate(np.abs(weighted_components_dica)), key=lambda x:np.abs(x[1]), reverse=True)]
        """

        def write_to_file(components, name):
            os.makedirs("{}/Components/{}".format(cancer_type, name), exist_ok=True)
            with open("{}/Components/{}/{}_{}_component_{}.csv".format(cancer_type, name, fname_base, name, n), 'w') as csv_file:
                with open("{}/Components/{}/{}_{}_component_{}.tsv".format(cancer_type, name, fname_base, name, n), 'w') as tsv_file:
                    for (idx, magnitude) in components:
                        print("{},{}".format(transcript_names[idx].gene_name, magnitude), file=csv_file)
                        print("{}\t{}".format(transcript_names[idx].gene_name, magnitude), file=tsv_file)

        if write_enrichment:
            for i, comps in enumerate(sorted_components):
                write_to_file(comps, reduced[i][-1])
            """
            write_to_file(weighted_components_pca, "PCA")
            write_to_file(weighted_components_ica, "ICA")
            write_to_file(weighted_components_cpca, "cPCA")
            write_to_file(weighted_components_dpca, "dPCA")
            write_to_file(weighted_components_drpca, "drPCA")
            write_to_file(weighted_components_dspca, "dsPCA")
            write_to_file(weighted_components_lda, "LDA")
            write_to_file(weighted_components_dica, "dICA")
            write_to_file(weighted_components_pls, "PLS")
            """

        """
        counts_pca = generate_curve(sorted_pca)
        counts_rpca = generate_curve(sorted_rpca)
        counts_spca = generate_curve(sorted_spca)
        counts_ica = generate_curve(sorted_ica)
        counts_cpca = generate_curve(sorted_cpca)
        counts_lda = generate_curve(sorted_lda)
        counts_dpca = generate_curve(sorted_dpca)
        counts_drpca = generate_curve(sorted_drpca)
        counts_dspca = generate_curve(sorted_dspca)
        counts_dica = generate_curve(sorted_dica)
        counts_pls  = generate_curve(sorted_pls)
        """

    """
    fig = plt.figure()
    plt.title("{} Components 1-{}".format(cancer_type, n), fontsize=28)
    plt.plot(counts_pca, label='PCA', linestyle='dashed')
    plt.plot(counts_lda, label='LDA', linestyle='dashed')
    plt.plot(np.array(counts_cpca)+0.1, label="cPCA", linestyle='dashed')
    plt.plot(counts_dpca, label='dPCA', linestyle='dashed')
    plt.plot(counts_drpca, label='drPCA', linestyle='dashed')
    plt.plot(counts_dspca, label="dsPCA", linestyle='dashed')
    plt.xlabel("Variable Rank", fontsize=22)
    plt.ylabel("Number of Oncogenes", fontsize=22)
    plt.legend()
    plt.show()
    """
if plot_onc_selection:
    sorted_components = plot_list_top(1)
    plot_list_top(2)
    plot_list_top(3)
    plot_list_top(5)
    #plot_list_top(10)
    #plot_list_top(25)
    #plot_list_top(100)

    plot_list_all(10)


def experiment(n_components=5):
    helper = lambda reps, labels, name: print("{}: {:.3f}".format(name,
        silhouette_score(reps[:, :n_components], labels)))
    print("="*20)
    print("Silhouette Scores, Training Data:")
    for (train, test, comps, name) in reduced:
        helper(train, train_labels, name)
    """print("PCA: {:.3f}".format(silhouette_score(
        pca_train_reduced[:, :n_components], train_labels)))
    print("ICA: {:.3f}".format(silhouette_score(
        ica_train_reduced[:, :n_components], train_labels)))
    print("LDA:  {:.3f}".format(silhouette_score(
        lda_train_reduced[:, :n_components], train_labels)))
    print("cPCA: {:.3f}".format(silhouette_score(
        cpca_train_reduced[:, :n_components], train_labels)))
    print("dPCA: {:.3f}".format(silhouette_score(
        dpca_train_reduced[:, :n_components], train_labels)))
    print("drPCA: {:.3f}".format(silhouette_score(
        drpca_train_reduced[:, :n_components], train_labels)))
    print("dsPCA: {:.3f}".format(silhouette_score(
        dspca_train_reduced[:, :n_components], train_labels)))
    print("dICA: {:.3f}".format(silhouette_score(
        dica_train_reduced[:, :n_components], train_labels)))
    """
    print('-'*20)
    print("Silhouette Scores, Testing Data:")
    for (train, test, comps, name) in reduced:
        helper(test, test_labels, name)
    """
    print("PCA: {:.3f}".format(silhouette_score(
        pca_test_reduced[:, :n_components], test_labels)))
    print("ICA: {:.3f}".format(silhouette_score(
        ica_test_reduced[:, :n_components], test_labels)))
    print("LDA:  {:.3f}".format(silhouette_score(
        lda_test_reduced[:, :n_components], test_labels)))
    print("cPCA: {:.3f}".format(silhouette_score(
        cpca_test_reduced[:, :n_components], test_labels)))
    print("dPCA: {:.3f}".format(silhouette_score(
        dpca_test_reduced[:, :n_components], test_labels)))
    print("drPCA: {:.3f}".format(silhouette_score(
        drpca_test_reduced[:, :n_components], test_labels)))
    print("dsPCA: {:.3f}".format(silhouette_score(
        dspca_test_reduced[:, :n_components], test_labels)))
    print("dICA: {:.3f}".format(silhouette_score(
        dica_test_reduced[:, :n_components], test_labels)))
    """
    print("="*20)

    """
    for pca, cpca, lda, dpca, drpca, dspca in zip(
        sorted_pca[:10], sorted_cpca[:10], sorted_lda[:10], sorted_dpca[:10], sorted_drpca[:10], sorted_dspca[:10]):
        print("{}\t{}\t{}\t{}".format(
            transcript_names[pca].gene_name, transcript_names[cpca].gene_name,
            transcript_names[lda].gene_name, transcript_names[dpca].gene_name,
            transcript_names[drpca].gene_name,
            transcript_names[dspca].gene_name))
    """

    """
    sorted_components = [i[0] for i in sorted(enumerate(diff_components[1]), key=lambda x:x[1])]
    print("dPCA Component 2")
    for index in sorted_components[-10:]:
        print(transcript_names[index])
    """

    #for index in sorted_components[-10:]:
    #    print(transcript_names[index])

    #sorted_components = [i[0] for i in sorted(enumerate(drpca_evecs[1]), key=lambda x:x[1])]
    #print("drPCA Component 2")
    #for index in sorted_components[-10:]:
        #print(transcript_names[index])


    ### Survival Analysis
    from lifelines import CoxPHFitter
    import pandas as pd
    from lifelines import AalenAdditiveFitter
    from lifelines.utils import k_fold_cross_validation

    def survival(reps, calc_random, pfi_filename, n_components=100):
        pfi = np.load(pfi_filename)

        good_rows = np.min(pfi, axis=1) >= 0
        pfi = pfi[good_rows]
        reps = reps[good_rows]

        good_features = np.var(reps, axis=0) > 1e-3
        reps = reps[:, good_features]
        reps = reps[:, :n_components]
        pfi[:, 0] = 1-pfi[:, 0]
        pfi = np.hstack((pfi, reps))
        columns = ['E', 'T']
        columns.extend([i for i in range(reps.shape[1])])
        reps = normalize(reps, axis=0) # normalize the features

        n_train = int(pct_train*len(pfi))
        n_val   = int(0.0*len(pfi))
        train_dataset = pd.DataFrame(pfi[:n_train], columns=columns)
        val_dataset = pd.DataFrame(pfi[n_train:n_train+n_val], columns=columns)
        test_dataset = pd.DataFrame(pfi[n_train+n_val:], columns=columns)
        pfi_test = pfi[n_train+n_val:]

        """
        # Using Aalen's Additive model
        aaf = AalenAdditiveFitter(fit_intercept=True)
        aaf.fit(train_dataset, 'T', event_col='E')
        predictions_df = aaf.predict_survival_function(test_dataset)
        predictions = predictions_df.values
        time_labels = predictions_df.index.values
        fig = plt.figure()
        loss = 0.
        for i in range(2, predictions.shape[1]):
            plt.plot(predictions[:, i])
            for j, pred in enumerate(predictions[:, i]):
                if pfi_test[i, 0] == 0: # Censored
                    if pfi_test[i, 1] < time_labels[j]:
                        loss += (1. - pred)
                    else:
                        pass
                else: # Not Censored
                    if pfi_test[i, 1] < time_labels[j]:
                        loss += (pred)
                    else:
                        loss += (1 - pred)
        print("AAF Loss:{:.3f}".format(loss))
        #aaf.plot()
        #plt.show()
        """

        #scores = k_fold_cross_validation(cph, train_dataset, 'T', event_col='E', k=3)
        step_size = 1.0
        cph = CoxPHFitter()
        while step_size > 0.01:
            try:
                k_fold_cross_validation(cph, train_dataset, 'T',
                    event_col='E', k=3, fitter_kwargs={"step_size": step_size})
                #cph.fit(train_dataset, 'T', event_col='E', step_size=step_size,
                #    show_progress=False)
                break
            except ValueError:
                step_size /= 2.0

        predictions_df = cph.predict_survival_function(test_dataset)
        predictions = predictions_df.values
        time_labels = predictions_df.index.values

        """
        #print(predictions.shape)
        #fig = plt.figure()
        loss = 0.
        for j in range(0, predictions.shape[0]):
            for i in range(0, predictions.shape[1]):
                if pfi_test[i, 0] == 1:
                    if pfi_test[i, 1] < time_labels[j]:
                        loss += predictions[j, i]
                    else:
                        loss += 1 - predictions[j, i]
                elif (pfi_test[i, 0] == 0 and pfi_test[i, 1] > time_labels[j]):
                    loss += 1 - predictions[j, i]
        print("CPH Loss:{:.3f}".format(loss))
        #plt.show()
        """

        #random_accs = []
        def calculate_acc(predictions):
            #print(predictions.shape)
            accs_time = []
            auc = 0.
            for j in range(0, predictions.shape[0]):
                incorrect = 0.
                correct = 0
                for i in range(0, predictions.shape[1]):
                    if pfi_test[i, 0] == 1:
                        if pfi_test[i, 1] < time_labels[j]:
                            if predictions[j, i] < 0.5:
                                correct += 1
                            else:
                                incorrect += 1
                        else:
                            if predictions[j, i] < 0.5:
                                incorrect += 1
                            else:
                                correct += 1
                    elif (pfi_test[i, 0] == 0 and pfi_test[i, 1] > time_labels[j]):
                        if predictions[j, i] < 0.5:
                            incorrect += 1
                        else:
                            correct += 1
                acc = float(correct) / (correct + incorrect)
                #print("{:.3f}".format(acc), end=' ')
                auc += acc
                accs_time.append(acc)
            return accs_time, auc / len(accs_time)

            #random_pred = float(j) / predictions.shape[1]
            #random_pred = 1. - float(j)/predictions.shape[0]#np.max([1. - float(j)/predictions.shape[0], float(j)/predictions.shape[0]])
            #true_rate = np.sum(predictions[j, :])/float(len(predictions[j, :]))
            #random_accs.append(np.max([np.min(
            #    [random_pred,
            #    true_rate]), np.min(
            #    [1-random_pred,
            #    1-true_rate])]))

        #cph.plot()
        accs_time, auc = calculate_acc(predictions)
        if calc_random:
            random_accs = []
            random_aucs = []
            for iteration in range(10):
                random_preds = np.zeros_like(predictions)
                for i in range(predictions.shape[0]):
                    rate = np.mean(pfi[:n_train, 1] < time_labels[i])
                    for j in range(len(predictions[i])):
                        random_preds[i, j] = 1 - scipy.stats.mode(pfi[:n_train, 1] < time_labels[i])[0][0]#1 - np.random.binomial(1, rate)
                random_acc, random_auc = calculate_acc(random_preds)
                random_accs.append(random_acc)
                random_aucs.append(random_auc)
            random_accs = np.mean(np.array(random_accs), axis=0)
            random_auc = np.mean(np.array(random_aucs), axis=0)
            return accs_time, auc, time_labels, random_accs, random_auc
        else:
            return accs_time, auc, time_labels
    """
    for fname in ["{}cases_pfi.npy".format(data_dir),
    "{}cases_dfi.npy".format(data_dir), "{}cases_dss.npy".format(data_dir), "{}cases_os.npy".format(data_dir)]:
        accs_pca, auc_pca, time_labels, accs_random, auc_random = survival(reduced[:n_fg], True, fname, n_components)
        accs_cpca, auc_cpca, _ = survival(fg_cpca, False, fname, n_components)
        accs_dpca, auc_dpca, _ = survival(fg_diff_transformed, False, fname, n_components)
        accs_drpca, auc_drpca, _ = survival(fg_drpca, False, fname, n_components)
        accs_dspca, auc_dspca, _ = survival(fg_spca_transformed, False, fname, n_components)
        accs_lda, auc_lda, _   = survival(fg_lda_transformed, False, fname, n_components)
        accs_dica, auc_dica, _ = survival(fg_dica_transformed, False, fname, n_components)
        accs_ica, auc_ica, _   = survival(fg_ica_transformed, False, fname, n_components)

        print("="*5 + "Survival AUC ({})".format(fname) + "="*5)
        print("Random:{:.3f}".format(auc_random))
        print("PCA:{:.3f}".format(auc_pca))
        print("ICA:{:.3f}".format(auc_ica))
        print("cPCA:{:.3f}".format(auc_cpca))
        print("LDA:{:.3f}".format(auc_lda))
        print("dPCA:{:.3f}".format(auc_dpca))
        print("drPCA:{:.3f}".format(auc_drpca))
        print("dsPCA:{:.3f}".format(auc_dspca))
        print("dICA:{:.3f}".format(auc_dica))
    """

    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start= 2, stop = 15, num = 3)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 20, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    C_list = [0.01, 0.1, 0.5, 1., 2.5, 5.0, 10., 25.]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    lr_grid = {'C': C_list}
    def case_control(x_train, x_test, y_train, y_test, calc_random):
        #lr = LogisticRegression(penalty='l2', C=1.)
        rf = RandomForestClassifier(n_estimators=3)
        clf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
            n_iter=8, cv=2, verbose=2, n_jobs=-1)

        clf.fit(x_train, y_train)
        #cross_validate(clf, x_train, y_train, return_train_score=False)
        
        if not test_tissue_type and not test_case_control:
            binarize = lambda x: label_binarize(x, classes=[0, 1, 2, 3, 4])
            auc_train = np.mean([roc_auc_score(binarize(y_train)[:, i], binarize(clf.predict(x_train))[:, i]) for i in range(5)])
            auc = np.mean([roc_auc_score(binarize(y_test)[:, i], binarize(clf.predict(x_test))[:, i]) for i in range(5)])
        else:
            auc_train = roc_auc_score(y_train, clf.predict(x_train), average='micro')
            auc = roc_auc_score(y_test, clf.predict(x_test))
        acc = accuracy_score(y_test, clf.predict(x_test))

        if calc_random:
            random_preds = np.ones_like(y_test)
            acc_random = 0.5 #accuracy_score(y_test, random_preds)
            auc_random = 0.5 #roc_auc_score(y_test, random_preds)
            return acc, auc, auc_train, acc_random, auc_random
        else:
            return acc, auc, auc_train

    """
    x_pca = np.vstack((reduced[:n_fg, :n_components], reduced[n_fg:, :n_components]))
    x_ica = np.vstack((fg_ica_transformed[:, :n_components], bg_ica_transformed[:, :n_components]))
    x_lda = np.vstack((fg_lda_transformed[:, :n_components], bg_lda_transformed[:, :n_components]))
    x_cpca = np.vstack((fg_cpca[:, :n_components], bg_cpca[:, :n_components]))
    x_dpca = np.vstack((fg_diff_transformed[:, :n_components], bg_diff_transformed[:, :n_components]))
    x_drpca = np.vstack((fg_drpca[:, :n_components], bg_drpca[:, :n_components]))
    x_dspca = np.vstack((fg_spca_transformed[:, :n_components], bg_spca_transformed[:, :n_components]))
    x_dica  = np.vstack((fg_dica_transformed[:, :n_components], bg_dica_transformed[:, :n_components]))
    y = np.ravel(np.vstack((np.ones((n_fg, 1)), np.zeros((n_bg, 1)))))
    x_pca_train, x_pca_test, x_ica_train, x_ica_test, x_cpca_train, x_cpca_test, x_lda_train, x_lda_test, x_dpca_train, x_dpca_test, x_drpca_train, x_drpca_test, x_dspca_train, x_dspca_test, x_dica_train, x_dica_test, y_train, y_test = train_test_split(
        x_pca, x_ica, x_cpca, x_lda, x_dpca, x_drpca, x_dspca, x_dica, y, test_size=0.33)
    """
    if test_tissue_type:
        helper  = lambda train, test: case_control(train[:n_train_fg, :n_components], test[:n_test_fg, :n_components], train_cancer_types, test_cancer_types, True)
    elif test_case_control:
        helper  = lambda train, test: case_control(train[:, :n_components], test[:, :n_components], train_labels, test_labels, True)
    else:
        helper  = lambda train, test: case_control(train[:n_train_fg, :n_components], test[:n_test_fg, :n_components], train_stages, test_stages, True)

    accs = []
    aucs = []
    aucs_train = []
    for (train_reduced, test_reduced, comps, name) in reduced:
        my_accs, my_auc, my_auc_train, accs_random, auc_random = helper(train_reduced, test_reduced)
        accs.append(my_accs)
        aucs.append(my_auc)
        aucs_train.append(my_auc_train)

    """
    np.savez_compressed("{}{}_reps".format(data_dir, fname_base),
        pca_train_reduced, pca_test_reduced, rpca_train_reduced, rpca_test_reduced,
        spca_train_reduced, spca_test_reduced, ica_train_reduced, ica_test_reduced,
        lda_train_reduced, lda_test_reduced, cpca_train_reduced, cpca_test_reduced,
        dpca_train_reduced, dpca_test_reduced, drpca_train_reduced, drpca_test_reduced,
        dspca_train_reduced, dspca_test_reduced, dica_train_reduced, dica_test_reduced)
    """
    """
    accs_pca, auc_pca, auc_pca_train, accs_random, auc_random = helper(pca_train_reduced, pca_test_reduced)
    accs_rpca, auc_rpca, auc_rpca_train, _, _ = helper(rpca_train_reduced, rpca_test_reduced)
    accs_spca, auc_spca, auc_spca_train, _, _ = helper(spca_train_reduced, spca_test_reduced)
    accs_ica, auc_ica, auc_ica_train, _, _   = helper(ica_train_reduced, ica_test_reduced)
    accs_lda, auc_lda, auc_lda_train, _, _   = helper(lda_train_reduced, lda_test_reduced)
    accs_pls, auc_pls, auc_pls_train, _, _ = helper(plsr_train_reduced, plsr_test_reduced)
    accs_cpca, auc_cpca, auc_cpca_train, _, _ = helper(cpca_train_reduced, cpca_test_reduced)
    accs_dpca, auc_dpca, auc_dpca_train, _, _ = helper(dpca_train_reduced, dpca_test_reduced)
    accs_drpca, auc_drpca, auc_drpca_train, _, _ = helper(drpca_train_reduced, drpca_test_reduced)
    accs_dspca, auc_dspca, auc_dspca_train, _, _ = helper(dspca_train_reduced, dspca_test_reduced)
    accs_dica, auc_dica, auc_dica_train, _, _   = helper(dica_train_reduced, dica_test_reduced)
    """

    """
    x_pca_train = pca_train_reduced[:, :n_components]
    x_pca_test  = pca_test_reduced[:, :n_components]
    x_ica_train = ica_train_reduced[:, :n_components]
    x_ica_test  = ica_test_reduced[:, :n_components]
    x_cpca_train = cpca_train_reduced[:, :n_components]
    x_cpca_test  = cpca_test_reduced[:, :n_components]
    x_lda_train = lda_train_reduced[:, :n_components]
    x_lda_test  = lda_test_reduced[:, :n_components]
    x_dpca_train = dpca_train_reduced[:, :n_components]
    x_dpca_test  = dpca_test_reduced[:, :n_components]
    x_drpca_train = drpca_train_reduced[:, :n_components]
    x_drpca_test = drpca_test_reduced[:, :n_components]
    x_dspca_train = dspca_train_reduced[:, :n_components]
    x_dspca_test = dspca_test_reduced[:, :n_components]
    x_dica_train = dica_train_reduced[:, :n_components]
    x_dica_test  = dica_test_reduced[:, :n_components]

    accs_pca, auc_pca, auc_pca_train, accs_random, auc_random = case_control(x_pca_train, x_pca_test, y_train, y_test, True)
    accs_ica, auc_ica, auc_ica_train   = case_control(x_ica_train, x_ica_test, y_train, y_test, False)
    accs_lda, auc_lda, auc_lda_train   = case_control(x_lda_train, x_lda_test, y_train, y_test, False)
    accs_cpca, auc_cpca, auc_cpca_train = case_control(x_cpca_train, x_cpca_test, y_train, y_test, False)
    accs_dpca, auc_dpca, auc_dpca_train = case_control(x_dpca_train, x_dpca_test, y_train, y_test, False)
    accs_drpca, auc_drpca, auc_drpca_train = case_control(x_drpca_train, x_drpca_test, y_train, y_test, False)
    accs_dspca, auc_dspca, auc_dspca_train = case_control(x_dspca_train, x_dspca_test, y_train, y_test, False)
    accs_dica, auc_dica, auc_dica_train   = case_control(x_dica_train, x_dica_test, y_train, y_test, False)
    """

    print("="*20)
    print("Case/Control AUC, Training Data")
    helper = lambda name, auc: print("{}: {:.3f}".format(name, auc))
    for i, auc_train in enumerate(aucs_train):
        helper(reduced[i][-1], auc_train)
    """
    print("Random:{:.3f}".format(auc_random))
    print("PCA:{:.3f}".format(auc_pca_train))
    print("rPCA:{:.3f}".format(auc_rpca_train))
    print("sPCA:{:.3f}".format(auc_spca_train))
    print("ICA:{:.3f}".format(auc_ica_train))
    print("LDA:{:.3f}".format(auc_lda_train))
    print("PLS:{:.3f}".format(auc_pls_train))
    print("cPCA:{:.3f}".format(auc_cpca_train))
    print("dPCA:{:.3f}".format(auc_dpca_train))
    print("drPCA:{:.3f}".format(auc_drpca_train))
    print("dsPCA:{:.3f}".format(auc_dspca_train))
    print("dICA:{:.3f}".format(auc_dica_train))
    """
    print("-"*20)
    print("Case/Control AUC, Testing Data")
    for i, auc in enumerate(aucs):
        helper(reduced[i][-1], auc)
    """
    print("Random:{:.3f}".format(auc_random))
    print("PCA:{:.3f}".format(auc_pca))
    print("rPCA:{:.3f}".format(auc_rpca))
    print("sPCA:{:.3f}".format(auc_spca))
    print("ICA:{:.3f}".format(auc_ica))
    print("LDA:{:.3f}".format(auc_lda))
    print("PLS:{:.3f}".format(auc_pls))
    print("cPCA:{:.3f}".format(auc_cpca))
    print("dPCA:{:.3f}".format(auc_dpca))
    print("drPCA:{:.3f}".format(auc_drpca))
    print("dsPCA:{:.3f}".format(auc_dspca))
    print("dICA:{:.3f}".format(auc_dica))
    """
    print("="*20)

    return aucs

"""
aucs_random = []
aucs_pca = []
aucs_ica = []
aucs_lda = []
aucs_pls = []
aucs_cpca = []
aucs_dpca = []
aucs_drpca = []
aucs_dspca = []
aucs_dica = []
"""
n_components_list = list(range(1, max_n_components))
n_iters = 5
n_methods = len(reduced)
results = np.zeros((n_methods, len(n_components_list), n_iters))
for n_iter in range(n_iters):
    for i, n_components in enumerate(n_components_list):
        print("N Components: {}".format(n_components))
        results[:, i, n_iter] = np.array(experiment(n_components))
        """
        auc_random, auc_pca, auc_ica, auc_lda, auc_cpca, auc_dpca, auc_drpca, auc_dspca, auc_dica = experiment(n_components)
        aucs_random.append(auc_random)
        aucs_pca.append(auc_pca)
        aucs_ica.append(auc_ica)
        aucs_lda.append(auc_lda)
        aucs_cpca.append(auc_cpca)
        aucs_dpca.append(auc_dpca)
        aucs_drpca.append(auc_drpca)
        aucs_dspca.append(auc_dspca)
        aucs_dica.append(auc_dica)
        """
"""
print("="*20 + "\nMean AUCs\n" + "="*20)
print("Random:{:.3f}".format(np.mean(results)))
print("PCA:{:.3f}".format(np.mean(aucs_pca)))
print("ICA:{:.3f}".format(np.mean(aucs_ica)))
print("LDA:{:.3f}".format(np.mean(aucs_lda)))
print("cPCA:{:.3f}".format(np.mean(aucs_cpca)))
print("dPCA:{:.3f}".format(np.mean(aucs_dpca)))
print("drPCA:{:.3f}".format(np.mean(aucs_drpca)))
print("dsPCA:{:.3f}".format(np.mean(aucs_dspca)))
print("dICA:{:.3f}".format(np.mean(aucs_dica)))
"""

fig = plt.figure()
#n_components_list = np.tile(np.array(n_components), (n_iters))
print("="*20)
print("Mean Case/Control AUC, Testing Data")
#for i, name in enumerate(["Random", "PCA", "rPCA", "sPCA", "ICA", "LDA", "PLS", "cPCA", "dPCA", "dsPCA", "drPCA", "dICA"]):
for i, (train, test, comps, name) in enumerate(reduced):
    if name == "Random" or name == "dICA":
        continue
    plt.errorbar(n_components_list, np.mean(results[i], axis=1), yerr=np.std(results[i], axis=1), label=name)
    print("{}:{:.3f}".format(name, np.mean(results[i])))
print("="*20)
"""
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="Random")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="PCA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="ICA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="LDA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="cPCA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="dPCA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="drPCA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="dsPCA")
plt.errorbar(n_components_list, np.mean(results[0], axis=1), yerr=np.std(results[0], axis=1), label="dICA")
"""
plt.xlabel("Number of Components", fontsize=22)
plt.ylabel("AUCROC", fontsize=22)
plt.legend(fontsize=18)
#plt.show()
plt.savefig("{}/aucroc_{}.png".format(cancer_type, fname_base))
plt.show()
#accs_random = 0.5 + np.abs(np.array(list(range(len(accs_pca))))/float(len(accs_pca)) - 0.5)
"""
plt.plot(time_labels, accs_pca, label="PCA")
plt.plot(time_labels, accs_dpca, label="dPCA")
plt.plot(time_labels, accs_drpca, label="drPCA")
plt.plot(time_labels, accs_dspca, label="dsPCA")
"""

"""
fig = plt.figure()
plt.plot(accs_random, label="Random")
plt.plot(accs_pca, label="PCA")
plt.plot(accs_dpca, label="dPCA")
plt.plot(accs_drpca, label="drPCA")
plt.plot(accs_dspca, label="dsPCA")
plt.legend()
plt.title(cancer_type, fontsize=26)
plt.show()
"""

# Cluster and show differential survival??


#aaf.print_summary()
# Using Cox Proportional Hazards model
#cph = CoxPHFitter()
#cph.fit(dataset, 'T', event_col='E')

#cph.print_summary()
