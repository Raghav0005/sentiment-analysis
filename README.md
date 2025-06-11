# Sentiment Analysis Trial Task

## Description
A Python-based sentiment analysis tool that fetches news articles from various sources through Google News and analyzes their sentiment using VADER sentiment analyzer and TextBlob. This provides sentiment classification, visualization, and word cloud generation capabilities.
Google News consolidates varying news articles from many different sources, which makes it a great place for web scraping.

## Features
- 📰 Fetch news articles from Google News RSS feeds
- 🎯 Sentiment analysis using VADER sentiment analyzer
- 📊 Subjectivity analysis using TextBlob
- 📈 Visual sentiment distribution charts
- ☁️ Word cloud generation for different sentiment categories
- 💾 Export results to CSV format

## Setup and Run the Program

### Prerequisites

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Raghav0005/sentiment-analysis.git
cd Sentiment-Analysis
```

2. Install required dependencies:
```bash
conda create -n sentiment-trial-period
conda activate sentiment-trial-period
conda install pandas nltk textblob matplotlib wordcloud feedparser spacy
python -m spacy download en_core_web_sm
```

Note that NLTK data download is automatic on first run.

## Usage

### Basic Usage
Run the main script with a topic and number of headlines to analyze:
```bash
python3 main.py <topic name> <number_of_headlines>
```

Examples:
```bash
# Analyze 50 articles about artificial intelligence
python3 main.py "artificial intelligence" 50

# Analyze 100 articles about technology
python3 main.py technology 100

# Analyze 25 articles about climate change
python3 main.py "climate change" 25
```

### Getting Help
To see all available options and usage examples:
```bash
python3 main.py --help
```

## Project Structure
```
Sentiment-Analysis/
├── main.py                 # Main execution script
├── analyzer.py             # Sentiment analysis functions
├── visualize.py            # Visualization and plotting functions
├── fetchers/
│   └── fetch_news.py       # News fetching utilities
├── data/                   # Generated output files
│   ├── news_with_sentiment.csv
│   ├── sentiment_distribution.png
│   ├── subjectivity_distribution.png
│   └── *_wordcloud.png
└── README.md
```

## Output Files
The script generates several output files in the `data/` directory:
- `news_with_sentiment.csv`: Complete dataset with sentiment scores
- `sentiment_distribution.png`: Bar chart of sentiment categories
- `subjectivity_distribution.png`: Histogram of subjectivity scores
- `positive_wordcloud.png`: Word cloud of positive headlines
- `neutral_wordcloud.png`: Word cloud of neutral headlines
- `negative_wordcloud.png`: Word cloud of negative headlines

## API Reference

### analyzer.py
- `analyze_sentiment(df)`: Performs VADER sentiment analysis
- `enrich_analysis(df)`: Adds TextBlob subjectivity scores

### visualize.py
- `plot_distribution(df, save_path)`: Creates sentiment distribution chart
- `plot_subjectivity(df, save_path)`: Creates subjectivity histogram
- `generate_wordcloud(text, title, save_path)`: Generates word clouds

### fetchers/fetch_news.py
- `fetch_google_news(topic, max_result)`: Fetches articles from Google News

## Additional Thoughts

This task was a great introduction to what we'll be working on as a team throughout the next 8 months. It also nicely built on my previous team project at Wat.AI, involving Stock Portfolio Optimization with Reinforcement Learning; now, we're looking at live world-wide sentiments, and how it may correlate with the movement of stock prices.

As midterms are approaching, with more time, this can be further expanded to include some finetuned Huggingface models.
I'm looking forward to contribute to this project, especially on the AI/ML ends of it!


