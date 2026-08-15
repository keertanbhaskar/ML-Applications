from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load dataset
iris = load_iris()

# take 1st 2 features
X = iris.data[:,:2]

# take only first 2 classes
y = iris.target

X = X[y != 2]
y = y[y != 2]

# split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# create svm model
model = SVC(kernel='linear')

# train the model
model.fit(X_train,y_train)

# make prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
new_flower = [[3.5,3.5]]
prediction = model.predict(new_flower)

print('Prediction:',prediction[0])