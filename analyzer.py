import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

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

def enrich_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich sentiment analysis results with additional text analysis metrics.
    
    This function adds subjectivity scores to the DataFrame using TextBlob's 
    sentiment analysis. Subjectivity measures how subjective or objective 
    the text is, with 0 being objective and 1 being subjective.
    
    Args:
        df (pd.DataFrame): DataFrame containing news articles with a 'title' column
    
    Returns:
        pd.DataFrame: Original DataFrame with added column:
                     - subjectivity: TextBlob subjectivity score (0-1, where 0 is objective and 1 is subjective)
    """
    # subjectivity from TextBlob
    df['subjectivity'] = df['title'].apply(
        lambda txt: TextBlob(txt).sentiment.subjectivity
    )
    return df
