
from sklearn import tree
features = [[140,1],[130, 1],[150,0],[170,0],[180,0]]
labels = ["apple","apple","orange","orange","orange"]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
p=clf.predict([[120,0]])
print(p)