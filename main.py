import os
import argparse
import pandas as pd
from fetchers.fetch_news import fetch_google_news
from analyzer import analyze_sentiment, enrich_analysis
from visualize import plot_distribution, generate_wordcloud, plot_subjectivity

def main(topic=None, max_results=None):
    """
    Main function to execute the sentiment analysis pipeline.
    
    This function:
    1. Fetches news articles for a specified topic
    2. Analyzes sentiment of the headlines
    3. Saves results to CSV files
    4. Generates and saves visualizations (distribution plot and word clouds)
    
    Args:
        topic (str): The topic to search for news articles (default: "Liberals")
        max_results (int): Number of articles to fetch per source (default: 50)
    
    Returns:
        None: Executes the complete sentiment analysis workflow and saves results
    """
    # Use default values if not provided
    if topic is None:
        topic = "Liberals"
    if max_results is None:
        max_results = 50

    os.makedirs('data', exist_ok=True)

    fetch_functions = [fetch_google_news]

    dataframes = []
    for fetch in fetch_functions:
        try:
            df = fetch(topic, max_results)
            dataframes.append(df)
        except Exception as e:
            print(f"Error fetching from {fetch.__name__}: {e}")

    # Combine all data sources and analyze sentiment
    all_df = pd.concat(dataframes, ignore_index=True)
    all_df = analyze_sentiment(all_df)
    all_df = enrich_analysis(all_df)

    # Save the complete dataset with sentiment analysis
    all_df.to_csv('data/news_with_sentiment.csv', index=False)

    print("\nOverall Sentiment Counts:")
    print(all_df['sentiment'].value_counts(), "\n")
    print("Sentiment by Source:")
    print(all_df.groupby('source')['sentiment'].value_counts(), "\n")

    plot_distribution(all_df, 'data/sentiment_distribution.png')
    plot_subjectivity(all_df, 'data/subjectivity_distribution.png')

    for sentiment in ['Positive', 'Neutral', 'Negative']:
        titles = ' '.join(all_df[all_df.sentiment == sentiment].title)
        filename = f'data/{sentiment.lower()}_wordcloud.png'
        generate_wordcloud(titles, f"{sentiment} Headlines Word Cloud", save_path=filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis Tool for News Headlines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 main.py "artificial intelligence" 100
  python3 main.py technology 25
  python3 main.py "climate change" 75
        '''
    )
    
    parser.add_argument(
        'topic', 
        help='Topic to search for news articles (use quotes for multi-word topics)'
    )
    
    parser.add_argument(
        'max_results', 
        type=int, 
        help='Number of headlines to fetch and analyze'
    )
    
    args = parser.parse_args()
    
    print(f"Searching for: '{args.topic}'")
    print(f"Number of headlines: {args.max_results}")
    print("-" * 50)
    
    main(args.topic, args.max_results)
