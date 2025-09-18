# Semantic Book Recommender

An AI-powered book recommendation system that uses semantic search and emotion analysis to provide personalized book suggestions based on natural language descriptions.

## Features

- **Semantic Search**: Find books based on natural language descriptions rather than simple keyword matching
- **Emotion-Based Filtering**: Filter recommendations by emotional tone (Happy, Surprising, Angry, Suspenseful, Sad)
- **Category Filtering**: Filter books by genre categories (Fiction, Nonfiction, Children's Fiction, Children's Nonfiction)
- **Interactive Web Interface**: User-friendly Gradio dashboard for exploring recommendations
- **Visual Gallery**: Display book covers with titles, authors, and descriptions

## Architecture

The system consists of several components:

1. **Vector Search Engine** (`vector-search.py`): Creates embeddings from book descriptions using OpenAI embeddings and stores them in a Chroma vector database for semantic similarity search
2. **Text Classification** (`text-classification.py`): Categorizes books using zero-shot classification with Facebook's BART model
3. **Sentiment Analysis** (`sentiment-analysis.py`): Analyzes emotional content of book descriptions using DistilRoBERTa emotion classifier
4. **Gradio Dashboard** (`gradio-dashboard.py`): Provides an interactive web interface for searching and browsing recommendations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/tarek-gritli/semantic-book-recommender.git
cd semantic-book-recommender
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Data Requirements

The system expects the following CSV files:

- `books.csv`: Original book dataset
- `books_cleaned.csv`: Preprocessed book data with descriptions obtained through the [Semantic Book Recommender Kaggle notebook](https://www.kaggle.com/code/tarekgritli/semantic-book-recommender)
- `books_with_categories.csv`: Books with category classifications
- `books_with_emotions.csv`: Books with emotion analysis scores

## Usage

### Running the Web Interface

Launch the Gradio dashboard:

```bash
python gradio-dashboard.py
```

The interface will open in your browser where you can:

- Enter a natural language description of the type of book you're looking for
- Select a category filter (optional)
- Select an emotional tone filter (optional)
- View recommended books in a visual gallery

### Processing Pipeline

If you need to process new book data:

1. **Prepare vector embeddings**:

```bash
python vector-search.py
```

2. **Classify book categories**:

```bash
python text-classification.py
```

3. **Analyze emotional content**:

```bash
python sentiment-analysis.py
```

## Technical Stack

- **Language Models**:
  - OpenAI Embeddings for semantic search
  - Facebook BART for zero-shot classification
  - DistilRoBERTa for emotion analysis
- **Vector Database**: Chroma for efficient similarity search
- **ML Framework**: Transformers, LangChain
- **Web Interface**: Gradio
- **Data Processing**: Pandas, NumPy

## Project Structure

```
semantic-book-recommender/
├── gradio-dashboard.py       # Web interface
├── vector-search.py          # Embedding generation and search
├── text-classification.py    # Category classification
├── sentiment-analysis.py     # Emotion analysis
├── requirements.txt          # Python dependencies
├── books*.csv               # Book datasets
├── tagged_descriptions.txt  # Processed descriptions for embedding
└── .env                     # Environment variables (not in repo)
```

## Dependencies

Key dependencies include:

- `langchain` and `langchain-chroma` for vector search
- `transformers` for NLP models
- `gradio` for web interface
- `pandas` for data processing
- `openai` for embeddings

See `requirements.txt` for complete list.
