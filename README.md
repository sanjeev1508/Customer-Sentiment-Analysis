# 🚀 Amazon Product Review Analysis

## 📌 Overview
This project is an AI-driven system that scrapes, processes, and analyzes Amazon product reviews to extract meaningful insights. It leverages **sentiment analysis, topic modeling, and emotion detection** to provide a detailed understanding of customer feedback.

## 📜 Features
- **Scrape Amazon Reviews** using ScrapingDog API.
- **Sentiment Analysis** with VADER & TextBlob.
- **AdoreScore Prediction** using a trained regression model.
- **Topic Classification** with a pre-trained model.
- **Emotion Detection** through NLP techniques.
- **Data Visualization** via graphs, heatmaps, radar charts, and word clouds.
- **Real-time & Historical Analysis** of review trends.

## 📂 Workflow
1. **Data Preparation**
   - Scrape and clean Amazon reviews using **ScrapingDog API**.
   - Extract sentiment words and vectorize text.
   - Cluster emotions using KMeans & Word2Vec.
2. **Model Training**
   - Train models for AdoreScore prediction, topic classification, and emotion clustering.
3. **Deployment & Visualization**
   - Load trained models and analyze old/new data.
   - Generate interactive dashboards and summaries.

## 🛠 Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, TensorFlow, NLTK, TextBlob)
- **Streamlit** (for web-based UI)
- **ScrapingDog API** (for Amazon review scraping)
- **Matplotlib & Seaborn** (for data visualization)


## 🚀 Running the App
After training the model in the Jupyter file, deploy using the command:
```bash
streamlit run app1.py
```

## 📊 Example Output
- **Sentiment Analysis**: Positive, Neutral, Negative classification.
- **Topic Classification**: Categories like Quality, Delivery, Customer Support, etc.
- **Emotion Detection**: Identifies primary & secondary emotions.
- **AdoreScore**: Predicts a numerical sentiment score.

## 📌 ScrapingDog API Setup
The project uses **ScrapingDog API** to scrape Amazon product reviews efficiently. To use it, add your API key in the `.env` file:
```
SCRAPINGDOG_API_KEY=your_api_key_here
```
Ensure you have sufficient API credits for seamless review scraping.

## 🤝 Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with your improvements!

## 📜 License
This project is licensed under the MIT License.

## 📧 Contact
For any queries, reach out to [Mail Here!](mailto:sanjeevikumar15@gmail.com).
