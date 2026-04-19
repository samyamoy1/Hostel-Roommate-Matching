import numpy as  np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

student_dataset = [
[22,8],[23,7],[21,9],[22,6],[23,8],
[1,5],[2,4],[3,3],[0,6],[2,5],
[21,7],[22,9],[23,6],[21,8],[22,7],
[2,6],[3,5],[1,4],[0,5],[2,4],

[20,6],[19,5],[18,6],[20,7],[19,6],
[18,5],[20,6],[19,7],[18,6],[20,5],

[3,7],[2,8],[1,7],[0,8],[2,7],
[3,6],[1,8],[0,7],[2,6],[3,7],

[22,5],[23,6],[21,5],[22,6],[23,5],
[1,7],[2,8],[3,7],[0,8],[2,7]
]

X=np.array(student_dataset)
#created the model
kmeans=KMeans(n_clusters=2,random_state=42)

#fit
kmeans.fit(student_dataset)

#output
labels=kmeans.labels_
centroids=kmeans.cluster_centers_

print(labels)
print(centroids)

#plot
plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(centroids[:,0],centroids[:,1],marker="X",s=200)

plt.xlabel("sleep time")
plt.xlabel("clenliness")
plt.title("k-means clustering(Hostel students)")

plt.show()