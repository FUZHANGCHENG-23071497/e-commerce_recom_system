import streamlit as st
import pandas as pd
import numpy as np
import torch, utils

@st.cache_data
def load_ratings():
    #Ratings
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, encoding='latin-1', engine='python')
    ratings.columns = ['userId','movieId','rating','timestamp']
    return ratings


@st.cache_data
def load_movies():
    #Movies
    movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, encoding='latin-1', engine='python')
    movies.columns = ['movieId','title','genres']
    return movies

@st.cache_data
def load_users():
    #Users
    users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, encoding='latin-1',  engine='python')
    users.columns = ['userId','gender','age','occupation','zipCode']
    return users

ratings = load_ratings()
movies = load_movies()
users = load_users()


def preprocessed_data(ratings, users, movies):
    # movie preprocessing
    movie_records = movies.copy()
    movies['genres'] = movies.apply(lambda row : row['genres'].split("|")[0],axis=1)
    movies['movie_year'] = movies.apply(lambda row : int(row['title'].split("(")[-1][:-1]),axis=1)
    movies.drop(['title'],axis=1,inplace=True)

    # combine rating and movie
    rating_movie = pd.merge(ratings,movies,how='left',on="movieId")

    # user preprocessing
    users['gender'] = users['gender'].replace({'F':0,'M':1})
    users['age'] = users['age'].replace({1:0,18:1, 25:2, 35:3, 45:4, 50:5, 56:6 })
    users.drop(['zipCode'],axis=1,inplace=True)

    # combine into final dataframe
    final_df = pd.merge(rating_movie,users,how='left',on='userId')

    return final_df, movie_records

final_df, movie_records = preprocessed_data(ratings, users, movies)

#settings for the data
wide_cols = ['movie_year','gender','age', 'occupation','genres','userId','movieId']
embeddings_cols = [('genres',20), ('userId',100), ('movieId',100)]
continuous_cols = ["movie_year","gender","age","occupation"]
target = 'rating'

#split the data and generate the embeddings
def data_process(final_df, wide_cols, embeddings_cols, continuous_cols, target):
    data_processed = utils.data_processing(final_df, wide_cols, embeddings_cols, continuous_cols, target, scale=True)
    return data_processed

data_processed = data_process(final_df, wide_cols, embeddings_cols, continuous_cols, target)

#setup for model arguments
wide_dim = data_processed['train_dataset'].wide.shape[1]
deep_column_idx = data_processed['deep_column_idx']
embeddings_input= data_processed['embeddings_input']
encoding_dict   = data_processed['encoding_dict']
hidden_layers = [100,50]
dropout = [0.5,0.5]

model_args = {
    'wide_dim': wide_dim,
    'embeddings_input': embeddings_input,
    'continuous_cols': continuous_cols,
    'deep_column_idx': deep_column_idx,
    'hidden_layers': hidden_layers,
    'dropout': dropout,
    'encoding_dict': encoding_dict,  # Will be updated during loading
    'n_class': 1
}

@st.cache_resource
def load_models(model_args):
    loaded_model = utils.load_model(utils.NeuralNet, model_args, "./model/movie_recommendation_model.pth", utils.device)
    loaded_model.compile(optimizer='Adam')
    return  loaded_model

loaded_model = load_models(model_args)

#predict_user = 1000

#top_k_movies = utils.recommend_top_k_movies(predict_user, final_df, movie_records, loaded_model, wide_cols, embeddings_cols, continuous_cols, k = 10, search_term = None)




# Streamlit UI Setup
st.title("Movie Recommendation System")

# Streamlit Tabs
tabs = st.tabs(["Home", "Dashboard", "Documentation"])

with tabs[0]:
    # Home Tab: The introduction and user inputs for recommendations
    st.header("Welcome to the Movie Recommendation System")
    
    st.markdown("""
    This app recommends movies to users based on their previous interactions. 
    You can enter your user ID, search for specific movies by title or genre, 
    and get personalized movie recommendations. Choose the number of recommendations you want to receive.
    """)

    # User selection: Enter user ID
    user_id = st.number_input("Enter User ID", min_value=1, max_value=final_df['userId'].max(), value=1)

    # Search feature: User can enter a query
    search_term = st.text_input("Search for Movies (Title or Genre)", "")

    # Get movie recommendations
    k = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
    recommended_movies = utils.recommend_top_k_movies(
        predict_user=user_id,
        final_df=final_df,
        movie_records=movie_records,
        model=loaded_model,  # Replace with your actual model
        wide_cols=wide_cols,
        embeddings_cols=embeddings_cols,
        continuous_cols=continuous_cols,
        k=k,
        search_term=search_term
    )

    # Display Recommendations
    st.subheader(f"Top {k} Movie Recommendations for User {user_id}")
    if not recommended_movies.empty:
        # Add "No." column and set it as index
        recommended_movies["No."] = range(1, len(recommended_movies) + 1)
        recommended_movies = recommended_movies.set_index("No.")
        recommended_movies["rating"] = 0
        recommended_movies["comments"] = ""
        st.data_editor(recommended_movies[['title', 'genres', 'movie_year', 'rating', 'comments']],
                      column_config={
                        "rating": st.column_config.NumberColumn(
                            "Your rating",
                            help="How much do you like this movie (1-5)?",
                            min_value=1,
                            max_value=5,
                            step=1,
                            format="%d ⭐",
                        ),
                        "comments": "Comments",
                    },
                    disabled=["title", "genres", "movie_year"])

        # Combine all the text from the DataFrame into a single string
        text = ' '.join(recommended_movies['genres'])
        with st.container(border=True):
            # Add a button to enable/disable word cloud
            if st.button('Generate Word Cloud'):
                st.write("The word cloud is now visible!")
                st.pyplot(utils.generate_wordcloud(text))
            else:
                st.write("Click the button above to generate the word cloud.")
    else:
        st.write("No recommendations available for this user with the search query.")

with tabs[1]:
    # Dashboard Tab: Data visualization and exploration
    st.header("Data Dashboard")
    st.markdown("""
    This section provides visual insights into the movie dataset. 
    You can explore distribution of ratings, genres, and other useful metrics.
    """)

    # Genre Distribution
    st.subheader("Genre Distribution")
    genre_counts = movie_records['genres'].value_counts()
    st.bar_chart(genre_counts)

    # Ratings Distribution
    st.subheader("Ratings Distribution")
    rating_counts = final_df['rating'].value_counts().sort_index()
    st.line_chart(rating_counts)

    # Movie Year Distribution
    st.subheader("Movie Release Year Distribution")
    year_counts = movie_records['movie_year'].value_counts().sort_index()
    st.line_chart(year_counts)

    # Correlation between rating and movie year (example)
    st.subheader("Correlation between Movie Year and Rating")
    rating_by_year = final_df.groupby('movieId')['rating'].mean().reset_index()
    movie_year_ratings = rating_by_year.merge(movie_records[['movieId', 'movie_year']], on='movieId')
    movie_year_ratings = movie_year_ratings.groupby('movie_year')['rating'].mean().reset_index()
    st.line_chart(movie_year_ratings.set_index('movie_year'))

with tabs[2]:
    # Documentation Tab: Information about the system
    st.header("Documentation")
    
    st.markdown("""
    ## Overview
    The Movie Recommendation System is designed to provide personalized movie suggestions using collaborative filtering techniques. It leverages user interaction data and movie features to predict ratings for unrated movies and offer recommendations tailored to each user’s preferences.
    
    ## How it Works
    - **User Interaction**: The app tracks user behavior through ratings, movie preferences, and other interactions, which are then used to predict potential movie ratings.
    - **Recommendation Model**: The system utilizes a collaborative filtering approach to predict user ratings for movies they have not rated. By applying these predictions, the system can recommend the top-k movies based on user-specific data.
      
    ## Features
    - **User Input**: Users can input their unique ID and specify their preferences to receive personalized movie recommendations.
    - **Search Function**: Search movies by title or genre to explore and discover new options.
    - **Personalized Recommendations**: Top-k movie recommendations are dynamically generated based on predicted ratings specific to the user.
    - **Data Visualizations**: Visualize movie genre distributions, rating trends, and movie popularity to better understand the data.

    ### Key Improvements:
    1. **Links**: Included links to the dataset and GitHub repository for users to access the necessary resources.
    2. **Installation Instructions**: Provided step-by-step installation instructions to help users set up the app locally.
    3. **Recommendation Algorithm**: Explained the Wide & Deep learning model used for recommendations, emphasizing its effectiveness in capturing both general and personalized patterns.
    4. **System Performance**: Discussed performance metrics like accuracy and latency, which ensure that recommendations are both accurate and quick.
    5. **User Interaction**: Highlighted how users can interact with the system, including registration, movie rating, and real-time feedback.
    6. **Things to Note**: Added a section on data privacy, limitations, and system scalability to inform users of any potential issues. 
    """)
