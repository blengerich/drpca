import numpy as np
from contrastive import CPCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, FastICA
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import sys
import os
sys.path.append("../")
from rpca import R_pca
from supervisedPCA import supervised_pca
import time

K = 2

bg_cov = np.array([[1.5, 0.5],
                   [0.5, 0.5]])
fg_cov = np.array([[3.5, 0.5],
                   [0.5, 0.1]])

n_fg = 1000
n_bg = 1000

offset = np.array([-0.0, 2.])
background_data = np.random.multivariate_normal(np.zeros(K), bg_cov, size=n_bg)# + np.random.multivariate_normal(np.zeros(K), bg_cov, size=n_fg)
foreground_data = background_data + offset + np.random.multivariate_normal(
    np.zeros(K), fg_cov, size=n_fg)
all_data = np.vstack((foreground_data, background_data))
#background_data = np.random.multivariate_normal(np.ones(K)*10, np.eye(K), size=50)

noise_level = 0.4
if noise_level > 0:
    for i in range(np.min([n_fg, n_bg])):
        if np.random.binomial(1, noise_level):
            temp = background_data[i].copy()
            background_data[i] = foreground_data[i].copy()
            foreground_data[i] = temp

foreground_data[-1] = np.array([4, -1])

# Differential PCA
differential_matched = np.array([
    foreground_data[i] - background_data[i] for i in range(n_fg)])
differential_matched = np.vstack((differential_matched, np.zeros_like(differential_matched)))

n_unmatched = 2000
differential_unmatched = np.array([
    foreground_data[np.random.choice(n_fg)] - background_data[np.random.choice(n_bg)] for i in range(n_unmatched)])
differential_unmatched = np.vstack((differential_unmatched, np.zeros_like(differential_unmatched)))


annotate_sil = False
def fit_all():
    if K == 3:
        n_components = 2
    else:
        n_components = 1
    pca = PCA(n_components=n_components)
    measure_silhouette = lambda reps: silhouette_score(
        reps, np.ravel(np.vstack((np.ones((n_fg, 1)), np.zeros((n_bg, 1))))))


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
        R = np.cov(data, rowvar=False)
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        if numComponents is not None:
            evecs = evecs[:, :numComponents]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T, evals, evecs

    names = []
    fig = plt.figure()
    if K == 3:
        from mpl_toolkits.mplot3d import Axes3D
        #ax = fig.add_subplot(2,4,1, projection='3d')
        ax = plt.gca()
        ax.scatter(foreground_data[:, 0], foreground_data[:, 1], foreground_data[:, 2], marker='*', alpha=0.5)
        ax.scatter(background_data[:, 0], background_data[:, 1], background_data[:, 2], marker='+', alpha=0.5)
        ax.set_zticks([])
    else:
        #ax = fig.add_subplot(2,4,1)
        ax = plt.gca()
        ax.scatter(foreground_data[:, 0], foreground_data[:, 1], marker='*', alpha=0.5)
        ax.scatter(background_data[:, 0], background_data[:, 1], marker='+', alpha=0.5)
    #ax = plt.gca()
    #names.append("Foreground Data")
    #names.append("Background Data")
    #ax.legend(names)


    def get_annotate_loc(ax, data):
        if data[np.argmin(data[:, 0]), 1] < data[np.argmax(data[:, 0]), 1]: # angling up
            x_loc = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.7 + ax.get_xlim()[0]
            y_loc = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.1 + ax.get_ylim()[0]
        else:
            x_loc = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.2 + ax.get_xlim()[0]
            y_loc = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.1 + ax.get_ylim()[0]
        return [x_loc, y_loc]



    raw_silhouette = measure_silhouette(all_data)
    #x_loc = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.7 + ax.get_xlim()[0]
    #y_loc = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.1 + ax.get_ylim()[0]
    annotate_location = get_annotate_loc(ax, all_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(raw_silhouette), annotate_location)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("Raw Data")
    print("Raw Data Silhouette: {:.3f}".format(raw_silhouette))

    y_lims = ax.get_ylim()
    x_lims = ax.get_xlim()
    def set_ax_lims(ax):
        y_expand = (y_lims[1] - y_lims[0])*0.05
        x_expand = (x_lims[1] - x_lims[0])*0.05
        ax.set_ylim([y_lims[0] - y_expand, y_lims[1]+y_expand])
        ax.set_xlim([x_lims[0] - x_expand, x_lims[1]+x_expand])
    set_ax_lims(ax)

    # Normal PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(all_data)
    reduced = reduced.dot(pca.components_)
    normal_components = pca.components_
    #plt.subplot(2,4,2)
    fig = plt.figure()
    ax = plt.gca()
    names = []
    ax.scatter(reduced[:n_fg, 0], reduced[:n_fg, 1], marker='*')
    ax.scatter(reduced[n_fg:, 0], reduced[n_fg:, 1], marker='+')
    pca_silhouette = measure_silhouette(reduced)

    set_ax_lims(ax)
    annotate_location = get_annotate_loc(ax, reduced)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(pca_silhouette), annotate_location)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("PCA")
    print("PCA Silhouette: {:.3f}".format(pca_silhouette))
    """
    names = ["PCA FG", "PCA BG"]
    names.append("PCA FG")
    names.append("PCA BG")
    plt.plot([0, pca.components_[0][0]], [0, pca.components_[0][1]], color='red')
    names.append("PCA Ax")
    ax.legend(names)
    """



    # Contrastive PCA
    # CPCA doesn't do dim reduction.
    mdl = CPCA(n_components=2)
    #print(foreground_data.shape)
    #print(background_data.shape)
    # For some reason, CPCA returns the data as the same size as the input data.
    alpha = 0
    mdl.fit(foreground_data, background_data)
    fg_cpca = mdl.transform(foreground_data)[0]
    #print(fg_cpca.shape)
    #bg_cpca = pca.fit_transform(background_data).dot(pca.components_)

    pca.fit(mdl.fg_cov - alpha*mdl.bg_cov)
    fg_cpca = np.expand_dims(fg_cpca[:, 0], 1).dot(pca.components_)
    bg_cpca = pca.transform(background_data).dot(pca.components_)

    #pca_directions = pca.components_
    #pca_directions = np.array([np.array([1.0]), np.array([1.0])])
    #print(projected_data)
    #print(dir(mdl))
    #print(mdl.pca_directions())
    #print(mdl.get_bg())
    #print(mdl.get_pca_directions())
    #print(mdl.pca_directions)
    #print(mdl.fg)

    #print(mdl.get_bg())
    #print(projected_data)
    #print(projected_data)
    #fg_cpca = (projected_data[2][:, :n_components].dot(pca_directions)).T
    #bg_cpca = (projected_data[3][:, :n_components].dot(pca_directions)).T
    #fg_cpca = mdl.get_fg()#[:, 0]projected_data[0]
    #bg_cpca = mdl.get_bg()#[:, 0], 1).dot(pca_directions)
    #fg_cpca = pca.fit_transform(fg_cpca)
    #bg_cpca = pca.transform(bg_cpca)
    #fg_cpca = fg_cpca.dot(pca.components_)
    #bg_cpca = bg_cpca.dot(pca.components_)
    #print(fg_cpca.shape)
    #print(bg_cpca.shape)
    #fig = plt.figure()
    #print(fg_proj.shape)
    #plt.subplot(2,4,3)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_cpca[:, 0], fg_cpca[:, 1], marker='*')
    ax.scatter(bg_cpca[:, 0], bg_cpca[:, 1], marker='+')
    set_ax_lims(ax)

    ax.set_xticks([])
    ax.set_yticks([])
    cpca_data = np.vstack((fg_cpca, bg_cpca))
    cpca_silhouette = measure_silhouette(cpca_data)
    annotate_location = get_annotate_loc(ax, cpca_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(cpca_silhouette), annotate_location)
    plt.tight_layout()
    plt.savefig("cPCA")
    print("cPCA Silhouette: {:.3f}".format(cpca_silhouette))
    #names.append("cPCA FG")
    #names.append("cPCA BG")
    #names = ["cPCA FG", "cPCA BG"]
    #ax.legend(names)



    # RPCA
    L, S = R_pca(all_data).fit(max_iter=10000, iter_print=1000)
    rpca_components, rpca_evals, rpca_evecs = get_differential(L, n_components)

    fg_rpca = foreground_data.dot(rpca_evecs)
    fg_rpca = np.array([fg_rpca[i, 0]*rpca_evecs[:, 0] for i in range(len(fg_rpca))])
    bg_rpca = background_data.dot(rpca_evecs)
    bg_rpca = np.array([bg_rpca[i, 0]*rpca_evecs[:, 0] for i in range(len(bg_rpca))])

    #plt.subplot(2,4,4)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_rpca[:, 0], fg_rpca[:, 1], marker='*')
    ax.scatter(bg_rpca[:, 0], bg_rpca[:, 1], marker='+')
    set_ax_lims(ax)

    ax.set_xticks([])
    ax.set_yticks([])
    rpca_data = np.vstack((fg_rpca, bg_rpca))
    rpca_silhouette = measure_silhouette(rpca_data)
    if annotate_sil:
        annotate_location = get_annotate_loc(ax, rpca_data)
        ax.annotate("S: {:.3f}".format(rpca_silhouette), annotate_location)

    plt.tight_layout()
    plt.savefig("rPCA")
    print("rPCA Silhouette: {:.3f}".format(rpca_silhouette))


    print("Fitting CCA...", end='')
    t = time.time()
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=n_components, scale=True)
    cca.fit(all_data, np.vstack((np.ones((n_fg, 1)), np.zeros((n_bg, 1)))))
    cca_components = cca.x_weights_.T
    cca_all_data = cca.transform(all_data).dot(cca_components)#cca.predict(train_data)#.dot(cca_components)
    fg_cca = cca_all_data[:n_fg]
    bg_cca = cca_all_data[n_fg:]
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_cca[:, 0], fg_cca[:, 1], marker='*')
    ax.scatter(bg_cca[:, 0], bg_cca[:, 1], marker='+')

    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("CCA")


    # sPCA
    #plt.subplot(2, 4, 5)
    fig = plt.figure()
    ax = plt.gca()
    spca = SparsePCA(n_components=n_components, max_iter=1000, verbose=False,
        alpha=10., ridge_alpha=0.0)
    spca.fit(all_data)
    spca_components = spca.components_
    spca_all_data = spca.fit_transform(all_data).dot(spca_components)
    fg_spca = spca_all_data[:n_fg]
    bg_spca = spca_all_data[n_fg:]
    ax.scatter(fg_spca[:, 0], fg_spca[:, 1], marker='*')
    ax.scatter(bg_spca[:, 0], bg_spca[:, 1], marker='+')

    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    spca_data = np.vstack((fg_spca, bg_spca))
    spca_silhouette = measure_silhouette(spca_data)
    if annotate_sil:
        annotate_location = get_annotate_loc(ax, spca_data)
        ax.annotate("S: {:.3f}".format(spca_silhouette), annotate_location)
    plt.tight_layout()
    plt.savefig("sPCA")
    print("sPCA Silhouette: {:.3f}".format(spca_silhouette))


    # LDA
    #plt.subplot(2, 4, 5)
    fig = plt.figure()
    ax = plt.gca()
    t = time.time()
    lda = LDA(n_components=n_components)
    lda.fit(all_data, np.vstack((np.ones((n_fg, 1)), np.zeros((n_bg, 1)))))
    lda_all_data = lda.transform(all_data).dot(lda.scalings_.T)
    print("LDA took {:.3f} seconds".format(time.time() - t))
    fg_lda = lda_all_data[:n_fg]
    bg_lda = lda_all_data[n_fg:]
    ax.scatter(fg_lda[:, 0], fg_lda[:, 1], marker='*')
    ax.scatter(bg_lda[:, 0], bg_lda[:, 1], marker='+')

    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    lda_data = np.vstack((fg_lda, bg_lda))
    lda_silhouette = measure_silhouette(lda_data)
    annotate_location = get_annotate_loc(ax, lda_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(lda_silhouette), annotate_location)
    print("LDA Silhouette: {:.3f}".format(lda_silhouette))
    plt.tight_layout()
    plt.savefig("LDA")
    #print(lda.scalings_)


    # Supervised PCA
    sup_pca = supervised_pca.SupervisedPCAClassifier(n_components=n_components)
    sup_pca.fit(all_data, np.vstack((np.ones((n_fg, 1)), np.zeros((n_bg, 1)))))
    fg_sup_pca = sup_pca.get_transformed_data(foreground_data).dot(sup_pca.get_components())
    bg_sup_pca = sup_pca.get_transformed_data(background_data).dot(sup_pca.get_components())
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_sup_pca[:, 0], fg_sup_pca[:, 1], marker='*')
    ax.scatter(bg_sup_pca[:, 0], bg_sup_pca[:, 1], marker='+')
    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    sup_pca_data = np.vstack((fg_sup_pca, bg_sup_pca))
    sup_silhouette = measure_silhouette(sup_pca_data)
    annotate_location = get_annotate_loc(ax, sup_pca_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(sup_silhouette), annotate_location)
    print("SupPCA Silhouette: {:.3f}".format(sup_silhouette))
    plt.tight_layout()
    plt.savefig("supPCA")


    # PLSRegression
    from sklearn.cross_decomposition import PLSRegression
    plsr = PLSRegression(n_components=n_components, scale=False)
    plsr.fit(all_data, np.vstack((np.ones((n_fg, 1)), np.zeros((n_bg, 1)))))
    fg_plsr = plsr.x_scores_[:n_fg].dot(plsr.x_weights_.T)
    bg_plsr = plsr.x_scores_[n_fg:].dot(plsr.x_weights_.T)
    #print(plsr.x_scores_.shape)
    #print(plsr.x_weights_.shape)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_plsr[:, 0], fg_plsr[:, 1], marker='*')
    ax.scatter(bg_plsr[:, 0], bg_plsr[:, 1], marker='+')
    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("PLSR")
    #plsr_silhouette = measure_silhouette()

    # dPCA-Mean
    x = np.mean(foreground_data, axis=0) - np.mean(background_data, axis=0)
    pca = PCA(n_components=n_components)
    x = x.reshape((1, -1))
    pca.fit(np.vstack((x, np.zeros_like(x))))
    dpca_mean_components = pca.components_
    print(dpca_mean_components)
    dpca_mean_transformed = pca.transform(all_data).dot(dpca_mean_components)
    fg_dpca_mean = dpca_mean_transformed[:n_fg]
    bg_dpca_mean = dpca_mean_transformed[n_fg:]
    #plt.subplot(2, 4, 6)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_dpca_mean[:, 0], fg_dpca_mean[:, 1], marker='*')
    ax.scatter(bg_dpca_mean[:, 0], bg_dpca_mean[:, 1], marker='+')
    names = ["dPCA_mean FG", "dPCA_mean BG "]
    #ax.legend(names)

    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    dpca_mean_data = np.vstack((fg_dpca_mean, bg_dpca_mean))
    dpca_mean_silhouette = measure_silhouette(dpca_mean_data)
    annotate_location = get_annotate_loc(ax, dpca_mean_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dpca_mean_silhouette), annotate_location)
    plt.tight_layout()
    plt.savefig("dPCA-Mean")
    print("dPCA-mean Silhouette: {:.3f}".format(dpca_mean_silhouette))

    # dPCA
    pca.fit(differential_matched)
    dpca_components = pca.components_
    print(dpca_components)
    dpca_transformed = pca.transform(all_data).dot(dpca_components)
    fg_dpca = dpca_transformed[:n_fg]
    bg_dpca = dpca_transformed[n_fg:]
    #plt.subplot(2,4,7)
    fig = plt.figure()
    set_ax_lims(ax)
    ax = plt.gca()
    #fg_mapped = fg_dpca*dpca_components
    #bg_mapped = bg_dpca*dpca_components
    ax.scatter(fg_dpca[:, 0], fg_dpca[:, 1], marker='*')
    #ax.scatter(fg_diff_transformed[:, 0], fg_diff_transformed[:, 1], marker='+')
    ax.scatter(bg_dpca[:, 0], bg_dpca[:, 1], marker='+')
    #ax.scatter(bg_diff_transformed[:, 0], bg_diff_transformed[:, 1], marker='*')
    #names.append("dPCA FG")
    #names.append("dPCA BG")
    #names = ["dPCA FG", "dPCA BG"]
    #ax.legend(names)

    ax.set_xticks([])
    ax.set_yticks([])
    dpca_data = np.vstack((fg_dpca, bg_dpca))
    dpca_silhouette = measure_silhouette(dpca_data)
    annotate_location = get_annotate_loc(ax, dpca_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dpca_silhouette), annotate_location)
    plt.tight_layout()
    plt.savefig("dPCA-Matched")
    print("dPCA-Matched Silhouette: {:.3f}".format(dpca_silhouette))

    # dPCA
    pca.fit(differential_unmatched)
    dpca_components = pca.components_
    print(dpca_components)
    dpca_transformed = pca.transform(all_data).dot(dpca_components)
    fg_dpca = dpca_transformed[:n_fg]
    bg_dpca = dpca_transformed[n_fg:]
    #plt.subplot(2,4,7)
    fig = plt.figure()
    set_ax_lims(ax)
    ax = plt.gca()
    #fg_mapped = fg_dpca*dpca_components
    #bg_mapped = bg_dpca*dpca_components
    ax.scatter(fg_dpca[:, 0], fg_dpca[:, 1], marker='*')
    #ax.scatter(fg_diff_transformed[:, 0], fg_diff_transformed[:, 1], marker='+')
    ax.scatter(bg_dpca[:, 0], bg_dpca[:, 1], marker='+')
    #ax.scatter(bg_diff_transformed[:, 0], bg_diff_transformed[:, 1], marker='*')
    #names.append("dPCA FG")
    #names.append("dPCA BG")
    #names = ["dPCA FG", "dPCA BG"]
    #ax.legend(names)

    ax.set_xticks([])
    ax.set_yticks([])
    dpca_data = np.vstack((fg_dpca, bg_dpca))
    dpca_silhouette = measure_silhouette(dpca_data)
    annotate_location = get_annotate_loc(ax, dpca_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dpca_silhouette), annotate_location)
    plt.tight_layout()
    plt.savefig("dPCA-Unmatched")
    print("dPCA-Unmatched Silhouette: {:.3f}".format(dpca_silhouette))

    # drPCA
    t = time.time()
    rpca = R_pca(differential_matched)
    L, S = rpca.fit(max_iter=10000, iter_print=1000)
    drpca_components, drpca_evals, drpca_evecs = get_differential(L, n_components)
    fg_drpca = foreground_data.dot(drpca_evecs)
    fg_drpca = np.array([fg_drpca[i, 0]*drpca_evecs[:, 0] for i in range(len(fg_drpca))])
    bg_drpca = background_data.dot(drpca_evecs)
    bg_drpca = np.array([bg_drpca[i, 0]*drpca_evecs[:, 0] for i in range(len(bg_drpca))])
    print("drPCA took {:.3f} seconds".format(time.time() - t))
    #plt.subplot(2,4,7)
    fig = plt.figure()
    ax = plt.gca()
    set_ax_lims(ax)
    ax.scatter(fg_drpca[:, 0], fg_drpca[:, 1], marker='*')
    ax.scatter(bg_drpca[:, 0], bg_drpca[:, 1], marker='+')
    names = ["drPCA FG", "drPCA BG"]
    drpca_data = np.vstack((fg_drpca, bg_drpca))
    drpca_silhouette = measure_silhouette(drpca_data)
    annotate_location = get_annotate_loc(ax, drpca_data)
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(drpca_silhouette), annotate_location)
    print("drPCA-Matched Silhouette: {:.3f}".format(drpca_silhouette))
    plt.tight_layout()
    plt.savefig("drPCA-Matched")

    # drPCA
    t = time.time()
    rpca = R_pca(differential_unmatched)
    L, S = rpca.fit(max_iter=10000, iter_print=1000)
    drpca_components, drpca_evals, drpca_evecs = get_differential(L, n_components)
    fg_drpca = foreground_data.dot(drpca_evecs)
    fg_drpca = np.array([fg_drpca[i, 0]*drpca_evecs[:, 0] for i in range(len(fg_drpca))])
    bg_drpca = background_data.dot(drpca_evecs)
    bg_drpca = np.array([bg_drpca[i, 0]*drpca_evecs[:, 0] for i in range(len(bg_drpca))])
    print("drPCA took {:.3f} seconds".format(time.time() - t))
    #plt.subplot(2,4,7)
    fig = plt.figure()
    ax = plt.gca()
    set_ax_lims(ax)
    ax.scatter(fg_drpca[:, 0], fg_drpca[:, 1], marker='*')
    ax.scatter(bg_drpca[:, 0], bg_drpca[:, 1], marker='+')
    names = ["drPCA FG", "drPCA BG"]
    drpca_data = np.vstack((fg_drpca, bg_drpca))
    drpca_silhouette = measure_silhouette(drpca_data)
    annotate_location = get_annotate_loc(ax, drpca_data)
    ax.set_xticks([])
    ax.set_yticks([])
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(drpca_silhouette), annotate_location)
    print("drPCA-Unmatched Silhouette: {:.3f}".format(drpca_silhouette))
    plt.tight_layout()
    plt.savefig("drPCA-Unmatched")



    # dsPCA
    #plt.subplot(2, 4, 8)
    fig = plt.figure()
    ax = plt.gca()
    spca = SparsePCA(n_components=n_components, max_iter=1000, verbose=False,
        alpha=10., ridge_alpha=0.0)
    spca.fit(differential_matched)
    dspca_components = spca.components_
    dspca_all_data = spca.transform(all_data).dot(dspca_components)
    fg_dspca = dspca_all_data[:n_fg]
    bg_dspca = dspca_all_data[n_fg:]
    #fg_dspca = spca.transform(foreground_data, ridge_alpha=0.0).dot(dspca_components)
    #bg_dspca = spca.transform(background_data, ridge_alpha=0.0).dot(dspca_components)
    ax.scatter(fg_dspca[:, 0], fg_dspca[:, 1], marker='*')
    ax.scatter(bg_dspca[:, 0], bg_dspca[:, 1], marker='+')
    plt.tight_layout()
    plt.savefig("dsPCA-Matched")
    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    dspca_data = np.vstack((fg_dspca, bg_dspca))
    dspca_silhouette = measure_silhouette(dspca_data)
    annotate_location = get_annotate_loc(ax, dspca_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dspca_silhouette), annotate_location)
    print("dsPCA-Matched Silhouette: {:.3f}".format(dspca_silhouette))

    # dsPCA
    #plt.subplot(2, 4, 8)
    fig = plt.figure()
    ax = plt.gca()
    spca = SparsePCA(n_components=n_components, max_iter=1000, verbose=False,
        alpha=10., ridge_alpha=0.0)
    spca.fit(differential_unmatched)
    dspca_components = spca.components_
    dspca_all_data = spca.transform(all_data).dot(dspca_components)
    fg_dspca = dspca_all_data[:n_fg]
    bg_dspca = dspca_all_data[n_fg:]
    #fg_dspca = spca.transform(foreground_data, ridge_alpha=0.0).dot(dspca_components)
    #bg_dspca = spca.transform(background_data, ridge_alpha=0.0).dot(dspca_components)
    ax.scatter(fg_dspca[:, 0], fg_dspca[:, 1], marker='*')
    ax.scatter(bg_dspca[:, 0], bg_dspca[:, 1], marker='+')
    plt.tight_layout()
    plt.savefig("dsPCA-Unmatched")
    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    dspca_data = np.vstack((fg_dspca, bg_dspca))
    dspca_silhouette = measure_silhouette(dspca_data)
    annotate_location = get_annotate_loc(ax, dspca_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dspca_silhouette), annotate_location)
    print("dsPCA-Unmatched Silhouette: {:.3f}".format(dspca_silhouette))


    # ICA
    print("Fitting ICA...", end='')
    t = time.time()
    ica = FastICA(n_components=n_components, max_iter=1000)
    ica.fit(all_data)
    #print(ica.mixing_)
    print("Took {:.3f} seconds.".format(time.time() - t))
    fg_ica = ica.transform(foreground_data).dot(ica.mixing_.T)
    bg_ica = ica.transform(background_data).dot(ica.mixing_.T)
    print(fg_ica)
    print(bg_ica)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_ica[:, 0], fg_ica[:, 1], marker='*')
    ax.scatter(bg_ica[:, 0], bg_ica[:, 1], marker='+')
    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ica_data = np.vstack((fg_ica, bg_ica))
    ica_silhouette = measure_silhouette(ica_data)
    annotate_location = get_annotate_loc(ax, ica_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(ica_silhouette), annotate_location)
    plt.tight_layout()
    plt.savefig("ICA")
    print("ICA Silhouette: {:.3f}".format(ica_silhouette))

    # ICA
    print("Fitting dICA...", end='')
    t = time.time()
    dica = FastICA(n_components=n_components, max_iter=1000)
    dica.fit(differential_matched)
    print("Took {:.3f} seconds.".format(time.time() - t))
    fg_dica = dica.transform(foreground_data).dot(dica.mixing_.T)
    bg_dica = dica.transform(background_data).dot(dica.mixing_.T)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_dica[:, 0], fg_dica[:, 1], marker='*')
    ax.scatter(bg_dica[:, 0], bg_dica[:, 1], marker='+')

    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    dica_data = np.vstack((fg_dica, bg_dica))
    dica_silhouette = measure_silhouette(dica_data)
    annotate_location = get_annotate_loc(ax, dica_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dica_silhouette), annotate_location)
    print("dICA-Matched Silhouette: {:.3f}".format(dica_silhouette))
    plt.tight_layout()
    plt.savefig("dICA-Matched")


    # ICA
    print("Fitting dICA...", end='')
    t = time.time()
    dica = FastICA(n_components=n_components, max_iter=1000)
    dica.fit(differential_unmatched)
    print("Took {:.3f} seconds.".format(time.time() - t))
    print(dica.components_)
    fg_dica = dica.transform(foreground_data).dot(dica.mixing_.T)
    bg_dica = dica.transform(background_data).dot(dica.mixing_.T)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(fg_dica[:, 0], fg_dica[:, 1], marker='*')
    ax.scatter(bg_dica[:, 0], bg_dica[:, 1], marker='+')

    set_ax_lims(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    dica_data = np.vstack((fg_dica, bg_dica))
    dica_silhouette = measure_silhouette(dica_data)
    annotate_location = get_annotate_loc(ax, dica_data)
    if annotate_sil:
        ax.annotate("S: {:.3f}".format(dica_silhouette), annotate_location)
    print("dICA-Unmatched Silhouette: {:.3f}".format(dica_silhouette))
    plt.tight_layout()
    plt.savefig("dICA-Unmatched")
    #plt.scatter(reduced[:n_fg, 0], reduced[:n_fg, 1])
    #plt.scatter(reduced[n_fg:, 0], reduced[n_fg:, 1])
    #plt.scatter(foreground_data[:, 0], foreground_data[:, 1])
    #plt.scatter(background_data[:, 0], background_data[:, 1])
    #plt.scatter(fg_diff_transformed[:, 0], fg_diff_transformed[:, 1])
    #plt.scatter(bg_diff_transformed[:, 0], bg_diff_transformed[:, 1])
    #plt.legend(names)
    #plt.title("Toy Example of Differential PCA")

    #plt.suptitle(title)
    """
    plt.tight_layout()
    if "Unmatched" in title:
        plt.savefig("unmatched.png")
    else:
        plt.savefig("matched.png")
    """
    #plt.show()

"""
print("=====Matched=====")
fit_all(differential_matched, "Matched Samples")
print("=====Unmatched=====")
fit_all(differential_unmatched, "Unmatched Samples")
"""
fit_all()
