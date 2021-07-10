import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Import data
wine_data = pd.read_csv('data/winequality-white.csv', delimiter=';')
# Train data
X = wine_data.drop(['quality'], axis=1)
# Target data
y = wine_data['quality']
# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Model training
model = SVC()
model.fit(X_train, y_train)
# Serialization using Pickle
pickle.dump(model, open('model.pkl', 'wb'))

