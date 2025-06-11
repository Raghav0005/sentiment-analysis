import os
import pandas as pd
from fetchers.fetch_news import fetch_google_news
from analyzer import analyze_sentiment
from visualize import plot_distribution, generate_wordcloud

def main():
    """
    Main function to execute the sentiment analysis pipeline.
    
    This function:
    1. Fetches news articles for a specified topic
    2. Analyzes sentiment of the headlines
    3. Saves results to CSV files
    4. Generates and saves visualizations (distribution plot and word clouds)
    
    Args:
        None
    
    Returns:
        None: Executes the complete sentiment analysis workflow and saves results
    """
    topic = "Mark Carney"        # ← change to any topic
    max_results = 50          # ← number of items per source

    os.makedirs('data', exist_ok=True)

    fetch_functions = [fetch_google_news]

    dataframes = []
    for fetch in fetch_functions:
        try:
            df = fetch(topic, max_results)
            dataframes.append(df)
            source_label = df['source'].iloc[0].replace(' ', '_').lower()
            df.to_csv(f'data/{source_label}_raw.csv', index=False)
        except Exception as e:
            print(f"Error fetching from {fetch.__name__}: {e}")

    all_df = pd.concat(dataframes, ignore_index=True)
    all_df = analyze_sentiment(all_df)

    all_df.to_csv('data/all_sentiment.csv', index=False)

    print("\nOverall Sentiment Counts:")
    print(all_df['sentiment'].value_counts(), "\n")
    print("Sentiment by Source:")
    print(all_df.groupby('source')['sentiment'].value_counts(), "\n")

    plot_distribution(all_df, save_path='data/sentiment_distribution.png')
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        titles = ' '.join(all_df[all_df.sentiment == sentiment].title)
        filename = f'data/{sentiment.lower()}_wordcloud.png'
        generate_wordcloud(titles, f"{sentiment} Headlines Word Cloud", save_path=filename)

if __name__ == '__main__':
    main()
