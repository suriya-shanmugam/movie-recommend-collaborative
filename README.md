# Movie Recommendation System - CRISP-DM Methodology

## Documentation Link:
- **Movie Recommendation System - CRISP-DM Methodology** - Project Document - Movie Recommender : [Writer Link](https://docs.google.com/document/d/1oCY_NQgiWtln7Hn89CMRzrbXCM9-gFj5axtF2BCd-0M/edit?usp=sharing)

## Presentation Link:
- **Movie Recommendation System** - Presentation - Movie Recommender : [Presentation Link](https://gamma.app/docs/CRISP-DM-Methodology-for-Movie-Recommendation-rw0uazhkhtsd47t)


## GitHub Repositories and Links:
- **Model 1 - Collaborative Filtering**: [GitHub Link](https://github.com/suriya-shanmugam/movie-recommend-collaborative)
- **Model 2 - Content-Based Filtering**: [GitHub Link](https://github.com/suriya-shanmugam/movie-recommend-content-based)
- **Model 3 - EDA**: [GitHub Link](https://github.com/RM-RAMASAMY/CMPE-255-Project)
- **Model 3 - Movie Recommender**: [GitHub Link](https://github.com/Ronak-Malkan/Movie-Recommender)

## Colab Links:
- **Model 1** - Content-Based Filtering: [Colab Link](Content-Based-Filtering-Recommendation.ipynb)
- **Model 2** - Movie Recommendation: [Colab Link](Recommend_Movie.ipynb)

## Web Links:
- **Streamlit Model 1** - Collaborative Filtering: [Streamlit Link](https://movie-recommend-collaborative-suriya.streamlit.app/)
- **Streamlit Model 2** - Content-Based Filtering: [Streamlit Link](https://movie-recommend-content-based-suriya.streamlit.app/)

## Video Demo Link:
- **Movie Recommendar** - By Quantum Bots:  [Youtube Link](https://www.youtube.com/watch?v=6bP4EBKqwVI)

---

## 1. **Business Objective**

The objective is to build a movie recommendation system that improves user engagement by recommending movies based on user preferences and the preferences of similar users.

### Key Goals:
- Increase user retention and satisfaction by providing personalized movie suggestions.

### Success Criteria:
- **Accuracy**: Use RMSE or MAE to measure recommendation accuracy.
- **Scalability**: Handle large datasets efficiently.
- **User Experience**: Provide diverse, relevant, and timely recommendations.

---

## 2. **Data Understanding**

### Data Sources:
- **Ratings Dataset**: Contains ratings from users for movies (UserID, MovieID, Rating).
- **Movies Dataset**: Contains metadata (MovieID, Title, Genre).

### Insights:
- **Sparsity**: The dataset is sparse (~95% missing ratings), which is typical in collaborative filtering.
- **Cold-start Problem**: New users or movies may not have enough data for recommendations.

---

## 3. **Data Preparation**

### Key Steps:
- **Data Cleaning**: Remove duplicates and ensure consistency between datasets.
- **Feature Engineering**: Create a sparse user-item matrix and map userId/movieId to indices.
- **Handling Sparsity**: Use collaborative filtering (KNN) and Bayesian averages to handle missing data.

### Data Splitting:
- Split the data into training (80%) and testing (20%) sets.

---

## 4. **Modeling**

### Model Selection:
1. **K-Nearest Neighbors (KNN)**: Simple, works well for sparse data. Uses cosine similarity or Euclidean distance.
2. **Singular Value Decomposition (SVD)**: A matrix factorization technique that reduces dimensionality and handles sparse data well.

### Implementation Steps:
- Preprocess the data into a sparse matrix.
- Train models using KNN or SVD.
- Generate recommendations based on similarities.

---

## 5. **Evaluation**

### Metrics:
- **Accuracy**: Evaluate using RMSE or MAE.
- **Diversity**: Ensure recommendations include both popular and niche titles.

### Findings:
- **KNN**: Performs well with denser data but struggles with sparsity.
- **SVD**: Better scalability and improved accuracy for large datasets.

---

## 6. **Deployment**

- Deploy the models on **Streamlit** for real-time recommendations.
- Monitor performance regularly to ensure quality recommendations.
