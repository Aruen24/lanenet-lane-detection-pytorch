"""
LLE : Locally Linear Embedding
Refercences :
[1]周志华.机器学习[M].清华大学出版社,2016:425.
[2]http://scikit-learn.sourceforge.net/dev/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html

Author : Ggmatch
Date : 2019/5/14
"""
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# 制造样本
n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10

fig = plt.figure(figsize=(6, 4))  #画板
gs = fig.add_gridspec(1,2)  #共2副子图
ax1 = fig.add_subplot(gs[0,0], projection='3d')  #第一幅子图表示原始样本分布
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

# LLE降维
n_components = 2

t0 = time()  #计时开始
Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components).fit_transform(X)
t1 = time()  #计时结束
ax2 = fig.add_subplot(gs[0,1])
ax2.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)  #第2副子图表示降维后样本分布
ax2.set_title("LLE (%.2g sec)" % (t1 - t0))
ax2.xaxis.set_major_formatter(NullFormatter())
ax2.yaxis.set_major_formatter(NullFormatter())

plt.show()





