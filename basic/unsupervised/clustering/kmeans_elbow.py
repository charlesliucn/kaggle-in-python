# -*- coding:utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5,1.5,(2,10))
cluster2 = np.random.uniform(5.5,6.5,(2,10))
cluster3 = np.random.uniform(3.0,4.0,(2,10))
X = np.hstack((cluster1,cluster2,cluster3)).T
plt.scatter(X[:,0],X[:,1])
print(X.shape)

K = range(1,10)
meandistortions = []
for k in K:
	kmeans = KMeans(n_clusters = k)
	kmeans.fit(X)
	distortion = sum(np.min(cdist(X, kmeans.cluster_centers_,"euclidean"),axis = 1))/X.shape[0]
	meandistortions.append(distortion)
plt.figure()
plt.plot(K, meandistortions, "bx-")
plt.xlabel("Number of Clusters K")
plt.ylabel("Average Dispersion")
plt.title("Selecting K with the ELBOW method")
plt.show()