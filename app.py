
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import numpy as np

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

@st.cache_data
def load_data():
    dataset = pd.read_csv(
        "movies_metadata.csv",
        usecols=["original_title", "genres", "original_language"],
        low_memory=False,
        nrows=3000
    )

    dataset = dataset[dataset["original_language"] == "en"].copy()

    def clean_genres(text):
        try:
            genres = ast.literal_eval(text)
            return " ".join([genre["name"] for genre in genres])
        except:
            return ""

    dataset["genres"] = dataset["genres"].fillna("").apply(clean_genres)

    dataset.rename(columns={"original_title": "Title"}, inplace=True)

    dataset = dataset[["Title", "genres"]]

    dataset.dropna(subset=["Title"], inplace=True)
    dataset.drop_duplicates(subset=["Title"], inplace=True)

    return dataset

movies_df = load_data()

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["genres"])

similarity = cosine_similarity(tfidf_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=movies_df["Title"],
    columns=movies_df["Title"]
)

def movie_recommendation(movie_name, top_rec):
    if movie_name not in similarity_df.index:
        return None

    scores = similarity_df[movie_name].sort_values(ascending=False)

    recommendations = scores.iloc[1:top_rec + 1]

    recommendations = (recommendations * 100).round(2)

    return recommendations

st.title("🎬 Movie Recommendation System")

st.write(
    "Find movies with similar genres using TF-IDF Vectorizer and Cosine Similarity."
)

movie_list = sorted(similarity_df.index.tolist())

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

top_n = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Recommend"):
    result = movie_recommendation(selected_movie, top_n)

    if result is None:
        st.error("Movie not found!")
    else:
        st.subheader(
            f"Top {top_n} movies similar to '{selected_movie}'"
        )

        st.dataframe(
            result.to_frame(name="Similarity (%)"),
            use_container_width=True
        )

        st.bar_chart(result)

        st.success(
            f"Most similar movie: {result.index[0]}"
        )

with st.expander("🔍 Show Movie Dataset"):
    st.dataframe(
        movies_df.head(100),
        use_container_width=True
    )

st.markdown("---")
st.caption(
    "Built with Streamlit | Content-Based Movie Recommendation System"
)