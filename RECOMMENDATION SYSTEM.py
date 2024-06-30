import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and preprocess the dataset
# For demonstration purposes, we use a small sample dataset

# Sample movie ratings dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movie_id': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4],
    'rating': [5, 4, 3, 4, 5, 2, 2, 5, 3, 5, 4, 4]
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Create a pivot table (user-item matrix)
user_movie_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

print("User-Movie Rating Matrix:")
print(user_movie_matrix)

# Step 2: Apply SVD for collaborative filtering
# Perform matrix factorization using TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
latent_matrix = svd.fit_transform(user_movie_matrix)

print("\nLatent Features (User-Movie Matrix Factorization):")
print(latent_matrix)

# Reconstruct the user-movie matrix using latent features
reconstructed_matrix = np.dot(latent_matrix, svd.components_)

print("\nReconstructed User-Movie Matrix (Approximation):")
print(pd.DataFrame(reconstructed_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.columns))

# Step 3: Generate recommendations for a given user
def recommend_movies(user_id, user_movie_matrix, reconstructed_matrix, top_n=3):
    # Find the index of the user in the matrix
    user_index = user_movie_matrix.index.get_loc(user_id)

    # Get the reconstructed ratings for the user
    user_reconstructed_ratings = reconstructed_matrix[user_index]

    # Sort the movies based on the reconstructed ratings
    movie_indices = user_movie_matrix.columns
    sorted_indices = np.argsort(user_reconstructed_ratings)[::-1]

    # Recommend top_n movies that the user has not rated yet
    recommended_movies = []
    for idx in sorted_indices:
        if user_movie_matrix.iloc[user_index, idx] == 0:
            recommended_movies.append(movie_indices[idx])
            if len(recommended_movies) == top_n:
                break

    return recommended_movies

# Example: Recommend movies for user_id = 1
user_id = 1
recommended_movies = recommend_movies(user_id, user_movie_matrix, reconstructed_matrix)
print(f"\nRecommended Movies for User {user_id}: {recommended_movies}")
