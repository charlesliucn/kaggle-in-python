# -*- coding:utf-8 -*-

# 导入所需库
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 简单的数值
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X = np.array(list(zip(x1,x2))).reshape(len(x1),2)

# 作图
plt.subplot(321)
plt.xlim([0,10])
plt.ylim([0,10])
plt.title("Instances")
plt.scatter(x1, x2)

# 每个聚类作出一个图
colors = ["b","g","r","c","m","y","k","b"]
markers = ["o","s","D","v","^","p","*","+"]
clusters = [2,3,4,5,8]
subplot_counter = 1
sc_scores = []
for t in clusters:
	# 子图
	subplot_counter += 1
	plt.subplot(3,2,subplot_counter)
	# KMeans聚类
	kmeans_model = KMeans(n_clusters = t).fit(X)
	# 聚类后作图
	for i, k in enumerate(kmeans_model.labels_):
		plt.plot(x1[i],x2[i], color = colors[k],
			marker = markers[k], ls = "None")
	plt.xlim([0,10])
	plt.ylim([0,10])
	# 计算轮廓系数
	sc_score = silhouette_score(X, kmeans_model.labels_,
		metric = "euclidean")
	sc_scores.append(sc_score)
	plt.title("K = %s, silhouette coefficient: %0.03f" % (t, sc_score))
plt.show()
# 比较不同聚类数的轮廓系数
plt.figure()
plt.plot(clusters,sc_scores,"*-")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient Score")
plt.show()