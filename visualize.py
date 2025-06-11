import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_distribution(df, save_path=None):
    """
    Create a bar plot showing the distribution of sentiment categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment analysis results with 'sentiment' column
        save_path (str, optional): File path to save the plot. If None, plot is not saved. Defaults to None.
    
    Returns:
        None: Displays and optionally saves the sentiment distribution bar chart
    """
    counts = df['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
    ax = counts.plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.ylabel('Number of Headlines')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_subjectivity(df, save_path=None):
    """
    Create a histogram showing the distribution of subjectivity scores.
    
    This function generates a histogram visualization of subjectivity scores
    from TextBlob analysis, showing how objective vs subjective the headlines are.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment analysis results with 'subjectivity' column
        save_path (str, optional): File path to save the plot. If None, plot is not saved. Defaults to None.
    
    Returns:
        None: Displays and optionally saves the subjectivity distribution histogram
    """
    ax = df['subjectivity'].hist(bins=20)
    plt.title('Subjectivity Distribution')
    plt.xlabel('Subjectivity')
    plt.ylabel('Count')
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()


def generate_wordcloud(text: str, title: str, save_path=None):
    """
    Generate and display a word cloud from the given text.
    
    Args:
        text (str): Text content to generate word cloud from
        title (str): Title to display above the word cloud
        save_path (str, optional): File path to save the word cloud image. If None, image is not saved. Defaults to None.
    
    Returns:
        None: Displays and optionally saves the word cloud visualization
    """
    if not text.strip():
        return
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()