
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from google_play_scraper import search, reviews, Sort
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os

# Streamlit app configuration
st.set_page_config(page_title="App Review Analysis", layout="wide")
st.title("App Review Analysis Dashboard")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
app_name = st.sidebar.text_input("Enter the app name", value="", placeholder="e.g., Instagram or com.instagram.android")
data_quantity = st.sidebar.number_input("Number of reviews to fetch", min_value=1, max_value=100, value=20)
country_code = st.sidebar.text_input("Enter country code (e.g., US, UK)", value="UK", placeholder="e.g., US").strip().upper()
analyze_button = st.sidebar.button("Analyze Reviews", type="primary")
reset_button = st.sidebar.button("Reset", type="secondary")

# Function to get package ID
def get_package_id(app_name):
    results = search(app_name, lang='en', country='us')
    return results[0]['appId'] if results else None

# Function to save figure to PNG
def save_figure(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    return filename

# Main app logic
if analyze_button:
    if not app_name:
        st.error("Please enter an app name.")
    else:
        with st.spinner(f"Fetching up to {data_quantity} reviews for: {app_name} from {country_code}..."):
            package_id = get_package_id(app_name)

            if not package_id:
                st.error("App not found.")
            else:
                # Fetch reviews
                data, _ = reviews(
                    package_id,
                    lang='en',
                    country=country_code,
                    sort=Sort.NEWEST,
                    count=data_quantity
                )

                # Convert to DataFrame
                data = pd.json_normalize(data)

                # Sentiment analysis
                sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)
                data["content"] = data["content"].astype("str")
                data["result"] = data["content"].apply(lambda x: sentiment_analysis(x))
                data["sentiment"] = data["result"].apply(lambda x: x[0]["label"])
                data["sentiment_score"] = data["result"].apply(lambda x: x[0]["score"])
                data1 = data[["content", "appVersion", "score", "at", "sentiment", "sentiment_score"]]

                # Save CSV
                csv_buffer = io.StringIO()
                data1.to_csv(csv_buffer, index=False)
                st.download_button("Download CSV", csv_buffer.getvalue(), "application_reviews.csv", "text/csv", key="csv-download")

                # Create PDF in memory
                pdf_buffer = io.BytesIO()
                pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

                # Title Page
                pdf.setFont("Helvetica-Bold", 24)
                pdf.drawCentredString(300, 600, "App Review Analysis")
                pdf.setFont("Helvetica", 18)
                pdf.drawCentredString(300, 550, f"App Name: {app_name.upper()}")
                pdf.setFont("Helvetica", 14)
                pdf.drawCentredString(300, 500, f"Link: https://play.google.com/store/apps/details?id={package_id}")
                pdf.showPage()

                # Sentiment Proportion Chart
                st.subheader("Sentiment Proportion (Positive vs. Negative)")
                fig1, ax1 = plt.subplots()
                sentiment_counts = data1['sentiment'].value_counts()
                colors = ['#4CAF50', '#F44336']
                sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=colors, labels=sentiment_counts.index, ylabel='')
                st.pyplot(fig1)
                fig_path = save_figure(fig1, "sentiment_proportion.png")
                pdf.drawImage(fig_path, 50, 400, width=500, height=300)
                pdf.showPage()
                plt.close(fig1)

                # Sentiment by App Version
                st.subheader("Sentiment Distribution by App Version (Top 5 Versions)")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                top_versions = data1['appVersion'].value_counts().nlargest(5).index
                filtered_data = data1[data1['appVersion'].isin(top_versions)]
                grouped = filtered_data.groupby(['appVersion', 'sentiment']).size().unstack(fill_value=0)
                grouped = grouped.loc[top_versions]
                grouped.plot(kind='bar', stacked=False, color={'POSITIVE': '#4CAF50', 'NEGATIVE': '#F44336'}, ax=ax2)
                plt.title('')  # Title set in Streamlit
                plt.xlabel('App Version')
                plt.ylabel('Number of Reviews')
                plt.xticks(rotation=45)
                plt.legend(title='Sentiment')
                plt.tight_layout()
                st.pyplot(fig2)
                fig_path = save_figure(fig2, "sentiment_by_version.png")
                pdf.drawImage(fig_path, 50, 400, width=500, height=300)
                pdf.showPage()
                plt.close(fig2)

                # Word Cloud for Positive Reviews
                st.subheader("Word Cloud of Positive Reviews")
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                positive_reviews = data[data['sentiment'] == 'POSITIVE']['content'].dropna()
                if not positive_reviews.empty:
                    text = ' '.join(positive_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens', max_words=100).generate(text)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('')
                    plt.tight_layout()
                    st.pyplot(fig3)
                    fig_path = save_figure(fig3, "positive_wordcloud.png")
                    pdf.drawImage(fig_path, 50, 400, width=500, height=300)
                    pdf.showPage()
                else:
                    st.write("No positive reviews found.")
                plt.close(fig3)

                # Word Cloud for Negative Reviews
                st.subheader("Word Cloud of Negative Reviews")
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                negative_reviews = data[data['sentiment'] == 'NEGATIVE']['content'].dropna()
                if not negative_reviews.empty:
                    text = ' '.join(negative_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds', max_words=100).generate(text)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title('')
                    plt.tight_layout()
                    st.pyplot(fig4)
                    fig_path = save_figure(fig4, "negative_wordcloud.png")
                    pdf.drawImage(fig_path, 50, 400, width=500, height=300)
                    pdf.showPage()
                else:
                    st.write("No negative reviews found.")
                plt.close(fig4)

                # Average Rating by Sentiment
                st.subheader("Average Sentiment Score by Sentiment Type")
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                avg_scores = data.groupby('sentiment')['score'].mean().reset_index()
                bars = plt.bar(avg_scores['sentiment'], avg_scores['score'], color=['#4CAF50' if s == 'POSITIVE' else '#F44336' if s == 'NEGATIVE' else '#FFC107' for s in avg_scores['sentiment']])
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 3), ha='center', va='bottom')
                plt.title('')
                plt.xlabel('Sentiment')
                plt.ylabel('Average Score')
                st.pyplot(fig5)
                fig_path = save_figure(fig5, "avg_sentiment_score.png")
                pdf.drawImage(fig_path, 50, 400, width=500, height=300)
                pdf.showPage()
                plt.close(fig5)

                # Finalize PDF
                pdf.save()
                pdf_buffer.seek(0)
                st.download_button("Download PDF", pdf_buffer, "app_review_analysis.pdf", "application/pdf", key="pdf-download")

                # Clean up temporary files
                for file in ["sentiment_proportion.png", "sentiment_by_version.png", "positive_wordcloud.png", "negative_wordcloud.png", "avg_sentiment_score.png"]:
                    if os.path.exists(file):
                        os.remove(file)

if reset_button:
    st.rerun()


