
import pickle
import streamlit as st
from surprise import SVD
 
# Load the model and data from the pickle file
with open('66130701904recommendation_movie_svd.pkl', 'rb') as file:
svd_model, movie_ratings, movies = pickle.load(file)
 
# Streamlit app
st.title("Movie Recommendation System")
st.write("Get personalized movie recommendations based on your preferences!")
 
# Input for User ID
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
 
if st.button("Get Recommendations"):
# Fetch rated and unrated movies for the user
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
 
# Generate predictions for unrated movies
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

# Sort predictions by estimated rating in descending order
sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
 
# Get top 10 movie recommendations
top_recommendations = sorted_predictions[:10]
 
st.subheader(f"Top 10 movie recommendations for User {user_id}:")
for recommendation in top_recommendations:
movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0] st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")
 
