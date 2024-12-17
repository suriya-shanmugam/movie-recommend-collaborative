import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

# Display dataframes
st.title("Movie Recommender System")
st.subheader("Ratings Data")
st.dataframe(ratings.head())

st.subheader("Movies Data")
st.dataframe(movies.head())

# Calculate statistics
n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

# Display statistics
st.write(f"Number of ratings: {n_ratings}")
st.write(f"Number of unique movieId's: {n_movies}")
st.write(f"Number of unique users: {n_users}")
st.write(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
st.write(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")

user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
st.write(f"Mean number of ratings for a given user: {user_freq['n_ratings'].mean():.2f}.")

# Plot data
sns.set_style("whitegrid")
fig1, ax1 = plt.subplots(figsize=(14,5))
sns.countplot(x="rating", data=ratings, palette="viridis", ax=ax1)
ax1.set_title("Distribution of movie ratings")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(14,5))
sns.kdeplot(user_freq['n_ratings'], shade=True, legend=False, ax=ax2)
ax2.axvline(user_freq['n_ratings'].mean(), color="k", linestyle="--")
ax2.set_xlabel("# ratings per user")
ax2.set_ylabel("density")
ax2.set_title("Number of movies rated per user")
st.pyplot(fig2)

# Calculate Bayesian average ratings
mean_rating = ratings.groupby('movieId')[['rating']].mean()
C = mean_rating['rating'].count()
m = mean_rating['rating'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = mean_rating.merge(bayesian_avg_ratings, on='movieId')
movie_stats = movie_stats.merge(movies[['movieId', 'title']])

# Display top and bottom movies by Bayesian average rating
st.subheader("Top Movies by Bayesian Average Rating")
st.dataframe(movie_stats.sort_values('bayesian_avg', ascending=False).head())

st.subheader("Bottom Movies by Bayesian Average Rating")
st.dataframe(movie_stats.sort_values('bayesian_avg', ascending=True).head())

# Create sparse matrix for collaborative filtering
def create_X(df):
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
sparsity = X.count_nonzero()/(X.shape[0]*X.shape[1])
st.write(f"Matrix sparsity: {round(sparsity*100, 2)}%")

# Save the sparse matrix
save_npz('data/user_item_matrix.npz', X)

# Find similar movies function using kNN
def find_similar_movies(movie_id, X, k=10, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    
    neighbour_ids.pop(0)
    
    return neighbour_ids

# Search and display similar movies by title
movie_titles = dict(zip(movies['movieId'], movies['title']))
title_to_id = {v: k for k, v in movie_titles.items()}  # Reverse mapping from title to ID

selected_movie_title = st.selectbox("Select a Movie Title to find similar movies", movies['title'].unique())
selected_movie_id = title_to_id[selected_movie_title]  # Map title to ID

similar_ids_cosine = find_similar_movies(selected_movie_id, X)
similar_ids_euclidean = find_similar_movies(selected_movie_id, X, metric="euclidean")

st.subheader(f"Movies similar to '{selected_movie_title}' using Cosine Similarity:")
for i in similar_ids_cosine:
    st.write(movie_titles[i])

st.subheader(f"Movies similar to '{selected_movie_title}' using Euclidean Distance:")
for i in similar_ids_euclidean:
    st.write(movie_titles[i])
