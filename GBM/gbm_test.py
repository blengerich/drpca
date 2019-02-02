import numpy as np
from contrastive import CPCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import time

#K = 2

#cov = np.eye(K)
#cov[0, 1] = 1.0
#cov[1, 0] = 1.0
#foreground_data = np.random.multivariate_normal(np.zeros(K), cov, size=50) + np.random.multivariate_normal(np.zeros(K), np.eye(K)*0.1, size=50)
#background_data = foreground_data + np.array([0, 5]) + np.random.multivariate_normal(np.zeros(K), np.eye(K)*0.1, size=50)
#background_data = np.random.multivariate_normal(np.ones(K)*10, np.eye(K), size=50)

foreground_data = np.load("cases.npy")
background_data = np.load("controls.npy")
matches = np.load("matches.npy")
transcript_names = np.load("transcript_names.npy")
n_fg = len(foreground_data)
n_bg = len(background_data)


print("Calculating dPCA")
t = time.time()
# Differential PCA
"""
differential = np.array([foreground_data[i] - background_data[j]
    for j in np.random.choice(n_bg, 5) for i in range(n_fg)])
"""
differential = np.array([foreground_data[i] - background_data[j] for [j, i] in matches])
differential = np.vstack((differential, np.array([background_data[j] - foreground_data[i] for [j, i] in matches])))
print(differential.shape)
#differential = np.vstack((differential, 0*differential))
#    # This is annoying - necessary?
print(differential.shape)
pca = PCA(n_components=2)
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
#differential = np.vstack((differential, 0*differential))
#    # This is annoying - necessary?
pca = PCA(n_components=2)
pca.fit(differential)
del differential
fg_diff_transformed = pca.transform(foreground_data)

bg_diff_transformed = pca.transform(background_data)
diff_components = pca.components_

sorted_components = [i[0] for i in sorted(enumerate(diff_components[0]), key=lambda x:x[1])]
#print(transcript_names)
print("dPCA Component 1")
for index in sorted_components[-10:]:
    #print(index)
    print(transcript_names[index])


sorted_components = [i[0] for i in sorted(enumerate(diff_components[1]), key=lambda x:x[1])]
#print(transcript_names)
print("dPCA Component 2")
for index in sorted_components[-10:]:
    #print(index)
    print(transcript_names[index])
#print(pca.components_)
#plt.plot([0, pca.components_[0][0]], [0, pca.components_[0][1]], color='black')
#plt.scatter(pca.components_[0][0], pca.components_[0][1], color='black', marker='*')
#names = ["Diff Ax"]
print("Took {:.3f} seconds.".format(time.time() - t))
plt.subplot(2,1,1)
plt.scatter(fg_diff_transformed[:, 0], fg_diff_transformed[:, 1], marker='*')
plt.scatter(bg_diff_transformed[:, 0], bg_diff_transformed[:, 1], marker='+')
names = ["dPCA GBM", "dPCA Control"]
plt.legend(names)
plt.title("Differential PCA on GBM Gene Expression")


# Normal PCA
print("Calculating normal PCA")
t = time.time()
pca = PCA(n_components=2)
all_data = np.vstack((foreground_data, background_data))
reduced = pca.fit_transform(all_data)
#reduced = reduced * pca.components_
normal_components = pca.components_
#plt.plot([0, pca.components_[0][0]], [0, pca.components_[0][1]], color='red')
#names.append("PCA Ax")
print("Took {:.3f} seconds".format(time.time() - t))

sorted_components = [i[0] for i in sorted(enumerate(normal_components[0]), key=lambda x:x[1])]
#print(transcript_names)
print("PCA Component 1")
for index in sorted_components[-10:]:
    print(transcript_names[index])

sorted_components = [i[0] for i in sorted(enumerate(normal_components[1]), key=lambda x:x[1])]
#print(transcript_names)
print("PCA Component 2")
for index in sorted_components[-10:]:
    print(transcript_names[index])

#plt.scatter(foreground_data[:, 0], foreground_data[:, 1])
#plt.scatter(background_data[:, 0], background_data[:, 1])
#names.append("Data FG")
#names.append("Data BG")

plt.subplot(2,1,2)
plt.scatter(reduced[:n_fg, 0], reduced[:n_fg, 1], marker='*')
plt.scatter(reduced[n_fg:, 0], reduced[n_fg:, 1], marker='+')
#names.append("PCA BRCA")
#names.append("PCA Control")
names = ["PCA GBM", "PCA Control"]
plt.legend(names)
plt.title("PCA on GBM Gene Expression")

# Contrastive PCA
"""
print("Calculating cPCA")
t = time.time()
mdl = CPCA(n_components=2)
projected_data = mdl.fit_transform(foreground_data, background_data)
fg_proj = projected_data[2]
bg_proj = projected_data[3]
plt.scatter(fg_proj[:, 0], fg_proj[:, 1])
plt.scatter(bg_proj[:, 0], bg_proj[:, 1])
names.append("cPCA BRCA")
names.append("cPCA Control")
print("Took {:.3f} seconds".format(time.time() - t))
"""

#plt.scatter(differential[:n_fg, 0], differential[:n_fg, 1])
#plt.scatter(differential[n_fg:, 0], differential[n_fg:, 1])
#names.append("Diff FG")
#fg_mapped = foreground_data*diff_components
#bg_mapped = background_data*diff_components

#names.append("dPCA BRCA")
#names.append("dPCA Control")
#names.append("Diff BG")


#plt.scatter(reduced[:n_fg, 0], reduced[:n_fg, 1])
#plt.scatter(reduced[n_fg:, 0], reduced[n_fg:, 1])
#plt.scatter(foreground_data[:, 0], foreground_data[:, 1])
#plt.scatter(background_data[:, 0], background_data[:, 1])
#plt.scatter(fg_diff_transformed[:, 0], fg_diff_transformed[:, 1])
#plt.scatter(bg_diff_transformed[:, 0], bg_diff_transformed[:, 1])

plt.show()