#import data set form scikit learn
from sklearn import datasets
iris = datasets.load_digits()

X= iris.data
Y= iris.target

# splitting the data into training data and testing datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .4)

#import classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

#giving data to classifier to train the model on that data.
my_classifier.fit(X_train,Y_train)

#giving the testing data set to test the model
predictions = my_classifier.predict(X_test)
print(predictions)
from sklearn.metrics import accuracy_score
test = accuracy_score(Y_test , predictions)
print(test)

