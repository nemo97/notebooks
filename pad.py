import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print "Predict sales of advartisement!"

data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col=0)

print data.head(10)

feature_cols=['TV','Radio','Newspaper']
X = data[feature_cols]
y = data['Sales']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

model = LinearRegression()
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print zip(y_test,y_pred)

print "Score = "
print model.score(X_test,y_test)


# import numpy as np
# from sklearn                        import metrics, svm
# from sklearn.linear_model           import LinearRegression
# from sklearn.linear_model           import LogisticRegression
# from sklearn.tree                   import DecisionTreeClassifier
# from sklearn.neighbors              import KNeighborsClassifier
# from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
# from sklearn.naive_bayes            import GaussianNB
# from sklearn.svm                    import SVC

# trainingData    = np.array([ [2.3, 4.3, 2.5],  [1.3, 5.2, 5.2],  [3.3, 2.9, 0.8],  [3.1, 4.3, 4.0]  ])
# trainingScores  = np.array( [3.4, 7.5, 4.5, 1.6] )
# predictionData  = np.array([ [2.5, 2.4, 2.7],  [2.7, 3.2, 1.2] ])

# clf = LinearRegression()
# clf.fit(trainingData, trainingScores)
# print("LinearRegression")
# print(clf.predict(predictionData))

# clf = svm.SVR()
# clf.fit(trainingData, trainingScores)
# print("SVR")
# print(clf.predict(predictionData))

# clf = LogisticRegression()
# clf.fit(trainingData, trainingScores)
# print("LogisticRegression")
# print(clf.predict(predictionData))

# clf = DecisionTreeClassifier()
# clf.fit(trainingData, trainingScores)
# print("DecisionTreeClassifier")
# print(clf.predict(predictionData))

# clf = KNeighborsClassifier()
# clf.fit(trainingData, trainingScores)
# print("KNeighborsClassifier")
# print(clf.predict(predictionData))

# clf = LinearDiscriminantAnalysis()
# clf.fit(trainingData, trainingScores)
# print("LinearDiscriminantAnalysis")
# print(clf.predict(predictionData))

# clf = GaussianNB()
# clf.fit(trainingData, trainingScores)
# print("GaussianNB")
# print(clf.predict(predictionData))

# clf = SVC()
# clf.fit(trainingData, trainingScores)
# print("SVC")
# print(clf.predict(predictionData))