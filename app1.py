import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import nltk
import requests
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sentence_transformers import SentenceTransformer

# Download required nltk data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("vader_lexicon", quiet=True)

# ScrapingDog API Key and URL (for new data)
API_KEY = "67bd841bc9066372a3d1e723"
SCRAPINGDOG_URL = "https://api.scrapingdog.com/amazon/reviews"

# File paths
ADORESCORE_MODEL_PICKLE = "adorescore_model.pkl"
TOPIC_MODEL_PICKLE = "topic_classifier.pkl"
SCRAPED_REVIEWS_CSV = "scraped_reviews.csv"
OLD_DATA_CSV = "updated_dataset.csv"  # contains at least "review" and "label" columns

SIA = SentimentIntensityAnalyzer()

@st.cache_resource
def load_models():
    with open(ADORESCORE_MODEL_PICKLE, "rb") as f:
        adorescore_data = pickle.load(f)
    regressor = adorescore_data["regressor"]
    embed_model = adorescore_data["embedding_model"]

    with open(TOPIC_MODEL_PICKLE, "rb") as f:
        topic_model = pickle.load(f)

    return regressor, embed_model, topic_model

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    return " ".join(tokens)

def predict_adorescore(review_text: str, regressor, embed_model) -> float:
    cleaned = clean_text(review_text)
    X_emb = embed_model.encode([cleaned])
    score = regressor.predict(X_emb)[0]
    return score

def predict_topic(review_text: str, topic_model: dict) -> str:
    cleaned = clean_text(review_text)
    vec = topic_model["vectorizer"].transform([cleaned])
    pred = topic_model["classifier"].predict(vec)[0]
    topic_label = topic_model["label_encoder"].inverse_transform([pred])[0]
    return topic_label

def predict_emotions(review_text: str) -> tuple:
    """
    Returns:
      primary_emotion, primary_intensity, secondary_emotion, secondary_intensity
    """
    sentiment = TextBlob(review_text).sentiment.polarity
    primary_emotion, secondary_emotion = "neutral", "neutral"
    primary_intensity, secondary_intensity = 0.0, 0.0

    if sentiment > 0.2:
        primary_emotion = "joy"
        primary_intensity = round(sentiment, 2)
    elif sentiment < -0.2:
        primary_emotion = "sadness"
        primary_intensity = round(abs(sentiment), 2)
    
    # For this example, we keep secondary emotion as neutral.
    return primary_emotion, primary_intensity, secondary_emotion, secondary_intensity

def determine_action_level(intensity: float) -> str:
    """
    Determines the action level based on primary intensity.
    """
    if intensity < 0.2:
        return "Low"
    elif 0.2 <= intensity < 0.5:
        return "Medium"
    else:
        return "High"

def scrape_amazon_reviews(asin: str):
    page = 1
    all_reviews = []
    while True:
        params = {"api_key": API_KEY, "asin": asin, "domain": "com", "page": str(page)}
        response = requests.get(SCRAPINGDOG_URL, params=params)
        if response.status_code != 200:
            break
        data = response.json()
        reviews = data.get("customer_reviews", [])
        if not reviews:
            break
        for review in reviews:
            all_reviews.append({
                "user": review.get("user", "Unknown"),
                "title": review.get("title", ""),
                "rating": review.get("rating", ""),
                "review_text": review.get("review", ""),
                "is_helpful": review.get("is_helpful", ""),
                "date": review.get("date", ""),
                "review_url": review.get("review_url", "")
            })
        page += 1
    
    df = pd.DataFrame(all_reviews)
    df.to_csv(SCRAPED_REVIEWS_CSV, index=False)
    return df if not df.empty else None

def generate_visuals(df):
    # Calculate sentiment values
    df["sentiment"] = df["review_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Emotion counts based on primary emotion
    emotions = ["joy", "sadness", "neutral"]
    emotion_counts = {e: df[df["primary_emotion"] == e].shape[0] for e in emotions}

    # Radar Chart for emotions
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(r=list(emotion_counts.values()), theta=emotions, fill='toself', name='Emotions'))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)

    # Sentiment Distribution Histogram
    fig_sentiment = px.histogram(df, x="sentiment", nbins=20, title="Sentiment Distribution", marginal="rug", color_discrete_sequence=["blue"])
    
    # Heatmap for Sentiment Scores
    fig_heatmap, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[["sentiment"]].T, cmap="coolwarm", annot=True, linewidths=0.5, cbar=True, ax=ax)
    st.pyplot(fig_heatmap)

    # Word Cloud of all reviews
    all_text = " ".join(df["review_text"])
    wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="coolwarm").generate(all_text)
    fig_wc, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)
    
    # Topic Distribution (if "predicted_topic" exists in the data)
    if "predicted_topic" in df.columns:
        topic_counts = df["predicted_topic"].value_counts()
        fig_topic = px.bar(topic_counts, x=topic_counts.index, y=topic_counts.values, 
                           title="Topic Distribution", 
                           labels={"x": "Topic", "y": "Count"}, 
                           color=topic_counts.index, 
                           color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig_topic = go.Figure()
        fig_topic.update_layout(title="Topic Distribution (Not Available)")

    # Pie Chart for Emotion Distribution
    fig_pie = px.pie(names=list(emotion_counts.keys()), 
                     values=list(emotion_counts.values()), 
                     title="Emotion Distribution",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    
    return fig_sentiment, radar_fig, fig_topic, fig_pie

def analyze_old_data(df, regressor, embed_model, topic_model):
    """
    Process old data:
      - Compute predicted adorescore, emotions, intensities, and action level.
      - Also, if desired, predict topic.
    The input df is assumed to have at least the column "review".
    """
    # Create a copy so as not to modify the original
    df = df.copy()
    
    # Predict AdoreScore
    df["predicted_adorescore"] = df["review"].apply(lambda x: predict_adorescore(x, regressor, embed_model))
    
    # Predict Topic (if needed)
    df["predicted_topic"] = df["review"].apply(lambda x: predict_topic(x, topic_model))
    
    # Predict Emotions and Intensities
    emotion_data = df["review"].apply(lambda x: pd.Series(predict_emotions(x)))
    emotion_data.columns = ["primary_emotion", "primary_intensity", "secondary_emotion", "secondary_intensity"]
    df = pd.concat([df, emotion_data], axis=1)
    
    # Determine Action Level
    df["action_level"] = df["primary_intensity"].apply(determine_action_level)
    
    return df

def overall_summary(df):
    """
    Returns a dictionary with overall (aggregated) values for the dataset.
    """
    summary = {}
    summary["Average AdoreScore"] = round(df["predicted_adorescore"].mean(), 2)
    summary["Average Primary Intensity"] = round(df["primary_intensity"].mean(), 2)
    summary["Average Secondary Intensity"] = round(df["secondary_intensity"].mean(), 2)
    summary["Most Common Primary Emotion"] = df["primary_emotion"].mode()[0] if not df["primary_emotion"].mode().empty else "N/A"
    summary["Most Common Action Level"] = df["action_level"].mode()[0] if not df["action_level"].mode().empty else "N/A"
    return summary

def main():
    st.title("Amazon Review Analyzer")
    st.write("Analyze reviews for AdoreScore, sentiment, topics, emotions, and action levels.")

    # Load models (used for both old and new data analysis)
    regressor, embed_model, topic_model = load_models()

    # Sidebar to choose analysis mode
    mode = st.sidebar.radio("Choose Analysis Mode", ("Analyze Old Data", "Scrape New Data and Analyze"))

    if mode == "Analyze Old Data":
        st.header("Analyzing Old Data")
        try:
            # Read the CSV containing old data (which includes at least "review" and "label")
            df_old = pd.read_csv(OLD_DATA_CSV)
            st.write("### Old Data Preview", df_old.head(15))

            # Process the data to compute metrics
            df_old_processed = analyze_old_data(df_old, regressor, embed_model, topic_model)

            # Display detailed analysis in table format.
            # Columns include: review, label, predicted adorescore, primary/secondary emotions, intensities, and action level.
            display_cols = ["review", "label", "predicted_adorescore", "primary_emotion", "primary_intensity",
                            "secondary_emotion", "secondary_intensity", "action_level", "predicted_topic"]
            st.write("### Detailed Analysis", df_old_processed[display_cols])
            
            # Compute and display overall summary values
            summary = overall_summary(df_old_processed)
            st.write("## Overall Summary")
            st.write(summary)
            
            # Generate and display additional visuals
            fig_sentiment, radar_fig, fig_topic, fig_pie = generate_visuals(df_old_processed.rename(columns={"review": "review_text"}))
            st.success("Analysis Complete!")
            st.plotly_chart(fig_sentiment)
            st.plotly_chart(radar_fig)
            st.plotly_chart(fig_topic)
            st.plotly_chart(fig_pie)
        except Exception as e:
            st.error(f"Error loading or analyzing old data: {e}")

    elif mode == "Scrape New Data and Analyze":
        st.header("Scraping New Data and Analyze")
        asin_input = st.text_input("Enter ASIN Code:")
        if st.button("Scrape Reviews"):
            if not asin_input.strip():
                st.error("Please enter a valid ASIN code.")
            else:
                with st.spinner("Fetching reviews..."):
                    reviews_df = scrape_amazon_reviews(asin_input)
                    if reviews_df is None:
                        st.error("No reviews found.")
                    else:
                        st.success("Reviews scraped successfully!")
                        st.write("### Scraped Reviews Preview", reviews_df.head())
        
        if st.button("Generate Insights from Scraped Data"):
            try:
                reviews_df = pd.read_csv(SCRAPED_REVIEWS_CSV)
                # For scraped data, we assume the review text is in "review_text" column.
                # We adjust the processing accordingly.
                reviews_df_processed = reviews_df.copy()
                reviews_df_processed["predicted_adorescore"] = reviews_df_processed["review_text"].apply(lambda x: predict_adorescore(x, regressor, embed_model))
                reviews_df_processed["predicted_topic"] = reviews_df_processed["review_text"].apply(lambda x: predict_topic(x, topic_model))
                emotion_data = reviews_df_processed["review_text"].apply(lambda x: pd.Series(predict_emotions(x)))
                emotion_data.columns = ["primary_emotion", "primary_intensity", "secondary_emotion", "secondary_intensity"]
                reviews_df_processed = pd.concat([reviews_df_processed, emotion_data], axis=1)
                reviews_df_processed["action_level"] = reviews_df_processed["primary_intensity"].apply(determine_action_level)
                
                # Display detailed analysis in table format.
                display_cols = ["user", "title", "rating", "review_text", "predicted_adorescore", 
                                "primary_emotion", "primary_intensity", "secondary_emotion", 
                                "secondary_intensity", "action_level", "predicted_topic"]
                st.write("### Detailed Analysis", reviews_df_processed[display_cols])
                
                # Compute and display overall summary values
                summary = overall_summary(reviews_df_processed)
                st.write("## Overall Summary")
                st.write(summary)
                
                # Generate additional visuals
                fig_sentiment, radar_fig, fig_topic, fig_pie = generate_visuals(reviews_df_processed)
                st.success("Analysis Complete!")
                st.plotly_chart(fig_sentiment)
                st.plotly_chart(radar_fig)
                st.plotly_chart(fig_topic)
                st.plotly_chart(fig_pie)
            except Exception as e:
                st.error(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
