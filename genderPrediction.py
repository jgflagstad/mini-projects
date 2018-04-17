from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

clfTree = tree.DecisionTreeClassifier()
clfKNN = KNeighborsClassifier(n_neighbors=3)
clfSVM = svm.SVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clfTree = clfTree.fit(X, Y)
clfKNN = clfKNN.fit(X, Y)
clfSVM = clfSVM.fit(X, Y)

predictionTree = clfTree.predict([[190, 70, 43]])
predictionKNN = clfKNN.predict([[190, 70, 43]])
predictionSVM = clfSVM.predict([[190, 70, 43]])

print(predictionTree)
print(predictionKNN)
print(predictionSVM)