import streamlit as st
import pandas as pd
import pickle
import requests

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=35fdf6d48f3f59f80a8426b5389a097c&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w185/" + data['poster_path']

movie_list = pickle.load(open('movie_dict.pkl', 'rb'))
movie_list = pd.DataFrame(movie_list)
st.title('Movie Recommender System')
option=st.selectbox('Select a movie to get recommendations',movie_list['title'].values)


similarity= pickle.load(open('similarity.pkl','rb'))

def recommend(movie):
 movie_index= movie_list [movie_list ['title']==movie].index[0]
 distance = similarity[movie_index]
 sorted_movies = sorted(list(enumerate(distance)),key=lambda x:x[1],reverse=True)

 recommended_movies=[]
 recommended_movies_poster=[]
 for i in sorted_movies[1:6]:
    movie_id=movie_list.iloc[i[0]].movie_id
    recommended_movies.append(movie_list.iloc[i[0]]['title'])
    posters=fetch_poster(movie_id)
    recommended_movies_poster.append(posters)
 return recommended_movies , recommended_movies_poster

if st.button('Recommend'):
    names,posters =recommend(option)



    if names and posters:
      col1, col2, col3,col4,col5 = st.columns(5)

      with col1:
       st.text(names[0])
       st.image(posters[0])

      with col2:
        st.text(names[1])
        st.image(posters[1])
      with col3:
        st.text(names[2])
        st.image(posters[2])

        with col4:
         st.text(names[3])
         st.image(posters[3])

         with col5:
          st.text(names[4])
          st.image(posters[4])
