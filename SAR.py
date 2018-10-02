import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import cv2
from PIL import Image

l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

# # #############################################################################
# # 4 circles
# img = circle1 + circle2 + circle3 + circle4

# # We use a mask that limits to the foreground: the problem that we are
# # interested in here is not separating the objects from the background,
# # but separating them one from the other.
# mask = img.astype(bool)

# img = img.astype(float)
# img += 1 + 0.2 * np.random.randn(*img.shape)

# # Convert the image into a graph with the value of the gradient on the
# # edges.
# graph = image.img_to_graph(img, mask=mask)

# # Take a decreasing function of the gradient: we take it weakly
# # dependent from the gradient the segmentation is close to a voronoi
# graph.data = np.exp(-graph.data / graph.data.std())

# # Force the solver to be arpack, since amg is numerically
# # unstable on this example
# labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
# label_im = np.full(mask.shape, -1.)
# label_im[mask] = labels

# plt.matshow(img)
# plt.matshow(label_im)

# # #############################################################################
# 2 circles
# img = circle1 + circle2
# img = cv2.imread("circle.jpg")
# img = io.imread('circle.jpg')
# print(img.shape)
# img = img[0:100][0:100]
# print(img.shape)
img = Image.open('circle.jpg')
img1 = img.resize((100, 100), Image.ANTIALIAS)
img1 = np.asarray(img1)
img1 = img1[:, :, 0]

img1 = img1.astype(float)
mask = img1.astype(bool)
print(img1.shape)

img1 += 1 + 0.2 * np.random.randn(*img1.shape)

graph = image.img_to_graph(img1, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=5, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img1)
plt.matshow(label_im)

plt.show()