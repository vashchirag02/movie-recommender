import numpy as Np 
import pandas as pd 

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

# print (movies.head())
# print (credits.head(1)['cast'].values)

df= movies.merge(credits,on='title')
# print(df.head())

# print(movies.info())
df=df[['movie_id','title','overview','genres','keywords','cast','crew']]
# print(df.head())

# print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

# print(df.iloc[0].genres)

import ast
def convert(obj):
  L=[]
  try:
   for i in ast.literal_eval(obj):
    L.append(i['name']) 
  except (ValueError, SyntaxError):
        return [] 
  return L

df['genres'] = df['genres'].apply(convert)
# print(df['genres'])
# print(df.iloc[:4].genres)

df['keywords'] = df['keywords'].apply(convert)                                                  
# print(df.iloc[:4].keywords)

# print(df.iloc[0].cast)

def convert2(obj):
    L=[]
    count=0
    try:
     for i in ast.literal_eval(obj):
      if count!=3:
        L.append(i['name']) 
        count+=1
      else:
        break
    except (ValueError, SyntaxError):
        return [] 
    return L

df['cast'] = df['cast'].apply(convert2)  
# print(df.iloc[:5].cast)

# print(df.iloc[0].crew)
def convert3(obj):
    L=[]
    try:
     for i in ast.literal_eval(obj):
       if i['job']=='Director':
          L.append(i['name'])
          break 
    except (ValueError, SyntaxError):
        return [] 
    return L

df['crew']=df['crew'].apply(convert3)
# print(df.iloc[:5].crew)

# print(df['overview'][0])
df['overview']=df['overview'].apply(lambda x:x.split())
# print(df['overview'][0])


df['genres']=df['genres'].apply(lambda x:[i.replace(" ","")for i in x])
df['cast']=df['cast'].apply(lambda x:[i.replace(" ","")for i in x])
df['crew']=df['crew'].apply(lambda x:[i.replace(" ","")for i in x])
df['keywords']=df['keywords'].apply(lambda x:[i.replace(" ","")for i in x])


df['tags']=df['overview']+df['genres']+df['keywords']+df['cast']+df['crew']

new_df=df[['movie_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
# print(new_df.head())


import nltk
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()
def stem(text):
    y=[]

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)
# print(new_df.head())

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
# print(vectors)

features= cv.get_feature_names_out()
# print(len(features))

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
# print(similarity.shape)

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distance = similarity[movie_index]
    movies=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies:
        print(new_df.iloc[i[0]].title)
        
    
recommend('Falcon Rising')
# sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

import pickle

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarity.pkl','wb'))

new_df['title'].values