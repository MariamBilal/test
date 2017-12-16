from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

#classifier function
class ScrappyKNN():
    def fit(self, X_train , Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def predict(self,X_test):
        predictions =[]
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
            
        return predictions
    def closest(self , row):
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            distance = euc(row , self.X_train[i])
            if distance < best_distance:
                best_distance = distance
                best_index = i
        return self.Y_train[best_index]    
        
    
#import data set form scikit learn
from sklearn import datasets
iris = datasets.load_iris()

X= iris.data
Y= iris.target

# splitting the data into training data and tetsting datasets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .4)


my_classifier = ScrappyKNN()

my_classifier.fit(X_train,Y_train)
predictions = my_classifier.predict(X_test)
print(predictions)
from sklearn.metrics import accuracy_score
test = accuracy_score(Y_test , predictions)
print(test)
