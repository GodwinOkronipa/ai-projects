#movie recommendation system using k-nearest neighbours and eucleadian distance (rt of the diff of lengts)
import numpy as np

# A dictionary representing users and their movie ratings (1-5 stars)
# The value 0 means the user has not rated that movie
# The order of movies is: "Inception", "Dune", "The Matrix", "Blade Runner"
ratings = {
    'User A': [5, 4, 0, 0],
    'User B': [4, 5, 0, 0],
    'User C': [0, 0, 5, 4],
    'User D': [0, 0, 4, 5],
    'User E': [5, 0, 0, 0] 
    #We will make a recommendation for user E
}

def euclidean_distance(user1_ratings, user2_ratings):
    """Calculates the Euclidean distance between two users based on their ratings."""
    # We only consider movies that *both* users have rated
    common_rated = [(r1, r2) for r1, r2 in zip(user1_ratings, user2_ratings) if r1 != 0 and r2 != 0]
    
    if not common_rated:
        return 0
    
    squared_diff = sum([(r1 - r2)**2 for r1, r2 in common_rated])
    return np.sqrt(squared_diff)

def get_recommendation(target_user, ratings_data, k=2):
    """
    Finds the k-nearest neighbors to the target user and recommends a movie.
    """
    distances = []
    
    # 1. Find the distance to every other user
    for user, user_ratings in ratings_data.items():
        if user == target_user:
            continue
        
        distance = euclidean_distance(ratings_data[target_user], user_ratings)
        distances.append((distance, user))
        
    # 2. Sort the users by distance and get the k-nearest neighbors
    distances.sort()
    neighbors = distances[1:k+1] # The first element is the distance to the user itself, so we skip it
    
    # Get the names of the neighbors and find their ratings
    neighbor_names = [user for distance, user in neighbors]
    
    # 3. Find movies the neighbors liked that the target user hasn't seen
    recommendations = {}
    target_ratings = ratings_data[target_user]
    
    for neighbor in neighbor_names:
        neighbor_ratings = ratings_data[neighbor]
        for i, rating in enumerate(neighbor_ratings):
            # Check if the neighbor liked the movie and the target user hasn't seen it
            if rating > 3 and target_ratings[i] == 0:
                movie_name = list(ratings_data['User A'].keys())[i]
                recommendations[movie_name] = recommendations.get(movie_name, 0) + 1
    
    # 4. Recommend the most frequently recommended movie
    if recommendations:
        return max(recommendations, key=recommendations.get)
    else:
        return "No new recommendations found."

# --- EXAMPLE USAGE ---

# The user we want to make a recommendation for
target_user = 'User E'

# Get the recommendation
recommended_movie = get_recommendation(target_user, ratings, k=2)

print(f"Based on your ratings, we recommend: {recommended_movie}")