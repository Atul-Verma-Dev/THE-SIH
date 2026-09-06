from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import pickle

df = pd.read_csv(r'C:\Users\akhil\py programs\THE SIH\marginalized_entrepreneurs_regression.csv')
df.info()
print(df.head())
print(df.shape)
print(df.isnull().sum())

df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['caste'] = df['caste'].map({'General': 0, 'OBC': 1, 'SC': 2, 'ST': 3})
df['location'] = df['location'].map({'urban': 0, 'rural': 1})

X = df[['income', 'gender', 'caste', 'disability', 'location','bpl_card']]
y = df['marginalization_score']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = ExtraTreesRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(mean_absolute_error(y_test,predictions))

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved! ✅")


