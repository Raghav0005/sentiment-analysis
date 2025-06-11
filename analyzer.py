import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment of news headlines using VADER sentiment analyzer.
    
    Args:
        df (pd.DataFrame): DataFrame containing news articles with a 'title' column
    
    Returns:
        pd.DataFrame: Original DataFrame with added columns:
                     - compound: VADER compound sentiment score (-1 to 1)
                     - sentiment: Categorized sentiment ('Positive', 'Negative', 'Neutral')
    """
    sid = SentimentIntensityAnalyzer()

    df['compound'] = df['title'].apply(lambda text: sid.polarity_scores(text)['compound'])

    def categorize(cmpd: float) -> str:
        """
        Categorize compound sentiment score into sentiment labels.
        
        Args:
            cmpd (float): VADER compound sentiment score (-1 to 1)
        
        Returns:
            str: Sentiment category ('Positive', 'Negative', or 'Neutral')
        """
        if cmpd >= 0.05:
            return 'Positive'
        elif cmpd <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment'] = df['compound'].apply(categorize)
    return df