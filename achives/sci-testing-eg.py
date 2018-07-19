from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
print('\n---------------\n', '\n---------------\n', iris , '\n---------------\n\n')
X, y = iris.data, iris.target
print('\n---------------\n', X.shape, '\n---------------\n', X , '\n---------------\n\n')
standardized_X = preprocessing.scale(X)
print('\n---------------\n', standardized_X.shape, '\n---------------\n', standardized_X , '\n---------------\n\n')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print('Score: ', metrics.accuracy_score(y, y_pred))