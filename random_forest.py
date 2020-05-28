#HEART DISEASE PREDICTION USING RANDOM FOEST CLASSIFICATION

#IMPORTING LIBRARIES
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#IMPORTING DATASET
df=pd.read_csv('heart.csv')
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

#SPLITTING INTO TRAIN AND TEST SET
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#FITTING RANDOM FOREST CLASSIFIER TO THE TRAINING SET
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#CHECKING THE ACCURACY
predictions = classifier.predict(X_test)
print("Accuracy score %f" % accuracy_score(y_test, predictions))