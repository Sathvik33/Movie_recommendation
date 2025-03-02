import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import os

csv_path = os.path.join(os.path.dirname(__file__), "data/imdb_top_1000.csv")
df = pd.read_csv(csv_path)

title_df = df[["Series_Title"]].copy()

df.drop(columns=["Poster_Link", 'Star1', 'Star2', 'Star3', 'Star4', 'Overview'], inplace=True)

df["Meta_score"] = df["Meta_score"].fillna(df["Meta_score"].mean())

df["Certificate"] = df["Certificate"].fillna(df["Certificate"].mode()[0])

df["Released_Year"] = df["Released_Year"].fillna(df["Released_Year"].median())

df["Runtime"] = df["Runtime"].str.removesuffix(" min").astype(float)

df["Gross"] = df["Gross"].astype(str)
df["Gross"] = df["Gross"].str.replace(",", "").replace("x", np.nan)
df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")
df["Gross"] = df["Gross"].fillna(df["Gross"].median())


string_cols = ['Certificate', 'Genre', 'Director']
num_cols = ['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
df_encoded = pd.get_dummies(df, columns=string_cols)
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
df_scaled = df_encoded.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='mean')
df_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns)


kmeans = KMeans(n_clusters=5, random_state=45)
clusters = kmeans.fit_predict(df_scaled)
df["clusters"] = clusters
title_df["clusters"] = clusters

st.title("🎬 Movie Recommendation System")

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "movie_name" not in st.session_state:
    st.session_state.movie_name = ""
if "show_genre_input" not in st.session_state:
    st.session_state.show_genre_input = False
if "genre_input" not in st.session_state:
    st.session_state.genre_input = ""

movie_name = st.text_input("Enter a movie name (only Hollywood):", value=st.session_state.movie_name)

def recommend_movies(movie_name, title_df, df, num_recommendations=5):
    if movie_name in title_df["Series_Title"].values:
        movie_cluster = title_df[title_df["Series_Title"] == movie_name]["clusters"].values[0]
        similar_movies = title_df[title_df["clusters"] == movie_cluster]
        similar_movies = similar_movies[similar_movies["Series_Title"] != movie_name]
        recommended = df[df["Series_Title"].isin(similar_movies["Series_Title"])]
        recommended = recommended.sort_values(by="IMDB_Rating", ascending=False)
        return recommended[["Series_Title", "IMDB_Rating", "Genre"]].head(num_recommendations)
    return None

def recommend_by_genre(genre_input, num_recommendations=5):
    genres = [g.strip().lower() for g in genre_input.split(",")]
    genre_movies = df[df["Genre"].apply(lambda x: any(g in x.lower() for g in genres) if isinstance(x, str) else False)]
    if genre_movies.empty:
        return None
    recommended = genre_movies.sort_values(by="IMDB_Rating", ascending=False)
    return recommended[["Series_Title", "IMDB_Rating", "Genre"]].head(num_recommendations)

if st.button("Get Recommendations"):
    if movie_name:
        recommendations = recommend_movies(movie_name, title_df, df)
        if recommendations is not None:
            st.session_state.recommendations = recommendations
            st.session_state.show_genre_input = False
        else:
            st.session_state.recommendations = None
            st.session_state.show_genre_input = True
    st.session_state.movie_name = movie_name

if st.session_state.show_genre_input:
    st.write("Movie not found in the dataset. You can get recommendations based on genres instead.")
    genre_input = st.text_input("Enter preferred genres (comma-separated):", value=st.session_state.genre_input)
    if st.button("Get Genre-Based Recommendations"):
        st.session_state.recommendations = recommend_by_genre(genre_input)
        st.session_state.genre_input = genre_input

if st.session_state.recommendations is not None:
    st.write("### Recommended Movies:")
    recommendations = st.session_state.recommendations.reset_index(drop=True)
    recommendations.index = range(1, len(recommendations) + 1)
    st.dataframe(recommendations)

if st.button("Clear"):
    st.session_state.recommendations = None
    st.session_state.movie_name = ""
    st.session_state.genre_input = ""
    st.session_state.show_genre_input = False
