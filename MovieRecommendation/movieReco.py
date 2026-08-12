from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib

df = pd.read_csv('movies.csv')
df1 = pd.read_csv('ratings.csv')

print("df shape:",df.shape)
print("df columns:",df.columns)

print("df1 shape:",df1.shape)
print('df1 columns:',df1.columns)

print(df.isnull().sum())
print(df1.isnull().sum())

print(df.head())
print(df1.head())

data = pd.merge(df,df1,on='movieId')
print(data.head(4))
print(data.columns)

movie_data = data.groupby(
  ['movieId','title','genres']
)['rating'].mean().reset_index()


# multi hot encoding
genres = movie_data['genres'].str.get_dummies(sep='|')
X = pd.concat([genres,movie_data[['rating']]],axis=1)


model = NearestNeighbors(n_neighbors=5)
model.fit(X)


# dump the model
joblib.dump(model,'movie_model.pkl')
joblib.dump(X,'movie features.pkl')
joblib.dump(movie_data,'movie_data.pkl')
