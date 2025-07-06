import streamlit as st
import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix, save_npz, load_npz
from implicit.als import AlternatingLeastSquares
import joblib

@st.cache_data
def load_data():
    ratings = pd.read_csv('data/ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestamp'],  engine='python')
    ratings.drop('timestamp', axis=1, inplace=True) 
    movies = pd.read_csv('data/movies.dat', sep='::', names=['movieId', 'title', 'genre'],  engine='python', encoding='latin1')
    movies.drop('genre', axis=1, inplace=True)
    return ratings, movies


@st.cache_resource
def load_models():
    loaded_model = joblib.load('models/als_model.joblib')
    loaded_matrix = load_npz('models/user_item_matrix.npz')
    
    return loaded_model, loaded_matrix

def main():
    st.title('Movie recommendations (ALS)')
    
    ratings, movies = load_data()
    model, user_item = load_models()

    user_ids = ratings['userId'].unique()
    selected_user = st.selectbox('Select user:', user_ids)

    st.subheader('User top rated movies')
    user_ratings = ratings[ratings['userId'] == selected_user]
    top_rated = user_ratings.sort_values('rating', ascending=False).head(5)
    top_rated = pd.merge(top_rated, movies, on='movieId')[['title', 'rating']]
    st.table(top_rated)

    st.subheader('Recommended Movies')
    recommended = model.recommend(selected_user, user_item[selected_user], N=10)[0]
    recommended_movies = []
    for item_id in recommended:
        title = movies[movies['movieId'] == item_id]['title'].values[0]
        recommended_movies.append({'title': title})
    
    st.table(pd.DataFrame(recommended_movies))

if __name__ == '__main__':
    main()