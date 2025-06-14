import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from google_play_scraper import app, reviews, Sort, search
from transformers import pipeline
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages
from google.colab import files

# Function to get package ID from app name
def get_package_id(app_name):
    results = search(app_name, lang='en', country='us')
    return results[0]['appId'] if results else None

# Streamlit UI
st.title("Google Play App Review Analyzer")

# User inputs
app_name = st.text_input("Enter the app name:", "play store")
data_quantity = st.number_input("How many reviews to fetch? (Sorted by newest)", min_value=1, value=20)
country_code = st.text_input("Enter country code (e.g., US, UK) [default = UK]:").strip().upper() if st.text_input else "UK"

package_id = get_package_id(app_name)

if st.button("Analyze Reviews"):
    if package_id:
        st.write(f"Fetching up to {data_quantity} reviews for: {app_name} ({package_id}) from country: {country_code}...")

        # Fetch reviews
        data, _ = reviews(
            package_id,
            lang='en',
            country=country_code,
            sort=Sort.NEWEST,
            count=data_quantity
        )
        data = pd.json_normalize(data)
        st.write("Sample Data:", data.head())

        # Sentiment analysis
        sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", truncation=True)
        data["content"] = data["content"].astype("str")
        data["result"] = data["content"].apply(lambda x: sentiment_analysis(x))
        data["sentiment"] = data["result"].apply(lambda x: x[0]["label"])
        data["sentiment_score"] = data["result"].apply(lambda x: x[0]["score"])
        data1 = data[["content", "appVersion", "score", "at", "sentiment", "sentiment_score"]]
        st.write("Processed Data:", data1)

        # Save and download CSV
        data1.to_csv("application_reviews.csv", index=False)
        with open("application_reviews.csv", "rb") as file:
            st.download_button(label="Download CSV", data=file, file_name="application_reviews.csv", mime="text/csv")

        # Generate PDF
        pdf = PdfPages("app_review_analysis.pdf")

        # Title Page
        fig0 = plt.figure(figsize=(8, 11))
        plt.text(0.5, 0.7, "App Review Analysis", ha='center', va='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.6, f"App Name: {app_name.upper()}", ha='center', va='center', fontsize=18)
        plt.text(0.5, 0.5, f"Link: https://play.google.com/store/apps/details?id={package_id}", ha='center', va='center', fontsize=14, wrap=True)
        plt.axis('off')
        pdf.savefig(fig0)
        plt.close(fig0)

        # Sentiment Proportion Chart
        fig1 = plt.figure()
        sentiment_counts = data1['sentiment'].value_counts()
        colors = ['#4CAF50', '#F44336']
        sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=colors, labels=sentiment_counts.index, title='Sentiment Proportion (Positive vs. Negative)', ylabel='')
        pdf.savefig(fig1)
        img = io.BytesIO()
        fig1.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        st.image(img, caption="Sentiment Proportion")
        plt.close(fig1)
        # Sentiment by App Version
        fig2 = plt.figure(figsize=(10, 6))
        top_versions = data1['appVersion'].value_counts().nlargest(5).index
        filtered_data = data1[data1['appVersion'].isin(top_versions)]
        grouped = filtered_data.groupby(['appVersion', 'sentiment']).size().unstack(fill_value=0)
        grouped = grouped.loc[top_versions]
        grouped.plot(kind='bar', stacked=False, color={'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336', 'NEUTRAL': '#FFC107'} if 'NEUTRAL' in grouped.columns else {'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336'}, ax=plt.gca())
        plt.title('Sentiment Distribution by App Version (Top 5 Versions)')
        plt.xlabel('App Version')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
        plt.tight_layout()
        pdf.savefig(fig2)
        img = io.BytesIO()
        fig2.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        st.image(img, caption="Sentiment by App Version")
        plt.close(fig2)

        # Word Cloud for Positive Reviews
        fig3 = plt.figure(figsize=(10, 5))
        positive_reviews = data[data['sentiment'] == 'POSITIVE']['content'].dropna()
        text = ' '.join(positive_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens', max_words=100).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Positive Reviews')
        plt.tight_layout()
        pdf.savefig(fig3)
        img = io.BytesIO()
        fig3.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        st.image(img, caption="Positive Reviews Word Cloud")
        plt.close(fig3)
        # Word Cloud for Negative Reviews
        fig4 = plt.figure(figsize=(10, 5))
        negative_reviews = data[data['sentiment'] == 'NEGATIVE']['content'].dropna()
        text = ' '.join(negative_reviews)
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds', max_words=100).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Negative Reviews')
        plt.tight_layout()
        pdf.savefig(fig4)
        img = io.BytesIO()
        fig4.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        st.image(img, caption="Negative Reviews Word Cloud")
        plt.close(fig4)
        # Average Rating by Sentiment
        fig5 = plt.figure(figsize=(8, 5))
        avg_scores = data.groupby('sentiment')['score'].mean().reset_index()
        bars = plt.bar(avg_scores['sentiment'], avg_scores['score'], color=['#4CAF50' if s == 'POSITIVE' else '#F44336' if s == 'NEGATIVE' else '#FFC107' for s in avg_scores['sentiment']])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 3), ha='center', va='bottom')
        plt.title('Average Sentiment Score by Sentiment Type')
        plt.xlabel('Sentiment')
        plt.ylabel('Average Score')
        pdf.savefig(fig5)
        img = io.BytesIO()
        fig5.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        st.image(img, caption="Average Score by Sentiment")
        plt.close(fig5)

        pdf.close()
        with open("app_review_analysis.pdf", "rb") as file:
            st.download_button(label="Download PDF", data=file, file_name="app_review_analysis.pdf", mime="application/pdf")

        st.success("Analysis complete!")
    else:
        st.error("App not found.")

# Note: Ensure all packages are installed in Colab before running