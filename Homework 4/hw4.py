import numpy as np
import scipy.io.arff
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

arff = scipy.io.arff

breast_data, breast_meta = arff.loadarff("breast.w.new.arff")

breast_targets = np.array(list(el["Class"] for el in breast_data))
breast_X = np.array(np.array(list(list(el[i] for i in range(9)) for el in breast_data)))

clusters2 = KMeans(n_clusters=2)
clusters3 = KMeans(n_clusters=3)

clusters2.fit(breast_X)
clusters3.fit(breast_X)

#4a
benign2 = [0, 0]
benign3 = [0, 0, 0]
malign2 = [0, 0]
malign3 = [0, 0, 0]

for i in range(len(clusters2.labels_)):
    if breast_targets[i] == b'benign':
        benign2[clusters2.labels_[i]] += 1
    else:
        malign2[clusters2.labels_[i]] += 1

for i in range(len(clusters3.labels_)):
    if breast_targets[i] == b'benign':
        benign3[clusters3.labels_[i]] += 1
    else:
        malign3[clusters3.labels_[i]] += 1
total_cluster_2_0 = benign2[0] + malign2[0]
total_cluster_2_1 = benign2[1] + malign2[1]

total_cluster_3_0 = benign3[0] + malign3[0]
total_cluster_3_1 = benign3[1] + malign3[1]
total_cluster_3_2 = benign3[2] + malign3[2]

ecr2 = 0.5 * ((total_cluster_2_0 - max(benign2[0], malign2[0])) + (total_cluster_2_1 - max(benign2[1], malign2[1])))
ecr3 = (1/3) * ((total_cluster_3_0 - max(benign3[0], malign3[0])) + (total_cluster_3_1 - max(benign3[1], malign3[1])) + (total_cluster_3_2 - max(benign3[2], malign3[2])))

print(ecr2, ecr3)

#4b
silhouette2 = silhouette_score(breast_X, clusters2.labels_)
silhouette3 = silhouette_score(breast_X, clusters3.labels_)

print(silhouette2, silhouette3)

#5
kbest = SelectKBest(mutual_info_classif, k=2).fit_transform(breast_X, breast_targets)

clusters = KMeans(n_clusters=3)
clusters.fit(kbest)

clusters_div = [[], [], []]

for i in range(len(clusters.labels_)):
    clusters_div[clusters.labels_[i]].append(kbest[i])

for i in range(len(clusters_div)):
    clusters_div[i] = np.array(clusters_div[i])

plt.scatter(clusters_div[0][:,0], clusters_div[0][:,1], color="red")
plt.scatter(clusters_div[1][:,0], clusters_div[1][:,1], color="green")
plt.scatter(clusters_div[2][:,0], clusters_div[2][:,1], color="blue")

plt.savefig("graphic.png", format="png")