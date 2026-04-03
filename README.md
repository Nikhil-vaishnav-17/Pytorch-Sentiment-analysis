# Movie Review Sentiment Analyzer

A PyTorch-powered sentiment analysis tool that classifies movie reviews as positive or negative using a BiLSTM neural network with GloVe embeddings. Built with Streamlit for an interactive web interface.

## Overview

This application combines a pre-trained BiLSTM model with SpaCy text preprocessing to analyze movie reviews in real-time. The system provides confidence scores and interpretive messages to help understand the model's predictions.

## Features

- **BiLSTM Neural Network**: 2-layer bidirectional LSTM for capturing context from both directions
- **GloVe Embeddings**: Pre-trained 100-dimensional word vectors for semantic understanding
- **SpaCy Preprocessing**: Lemmatization, tokenization, and punctuation removal
- **Interactive Streamlit UI**: Clean, responsive web interface for real-time analysis
- **Confidence Metrics**: Shows both percentage confidence and raw probability
- **Example Reviews**: Three pre-loaded examples (Positive, Negative, Mixed) for quick testing
- **Text Statistics**: Real-time word and character count
- **Detailed Results**: Expandable section showing raw model output and metrics

## Model Architecture

### Network Structure
- **Embedding Layer**: 15,000 vocabulary size, 100-dimensional GloVe embeddings (frozen)
- **Embedding Dropout**: 0.4
- **BiLSTM**: 2 stacked layers with 128 hidden units each, bidirectional processing
- **LSTM Dropout**: 0.3 between layers
- **Output Dropout**: 0.6 before classification
- **Linear Layer**: Maps 256-dimensional concatenated hidden states to single output
- **Activation**: Sigmoid for probability output (0-1 range)

### Key Hyperparameters
- Vocabulary size: 15,000 words
- Embedding dimension: 100
- Hidden dimension: 128
- Max sequence length: 200 tokens
- Classification threshold: 0.5 (Positive ≥ 0.5, Negative < 0.5)

### Model Performance
- Training accuracy: ~89%
- Test accuracy: ~87%
- Inference time: <100ms per review (CPU)

## Project Structure

```
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── training/
│   ├── preprocess.py              # Text preprocessing functions
│   ├── SentimentLSTM.py           # PyTorch model class
│   ├── Training.ipynb             # Model training notebook
│   ├── glove.6B.100d.txt          # GloVe embeddings (100-dimensional)
│   └── models/
│       ├── sentiment_model.pt     # Trained model weights
│       ├── vocab.pkl              # Vocabulary dictionary
│       └── model_config.json      # Model configuration
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or compatible package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Pytorch-Sentiment-analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy model** (required)
   ```bash
   python -m spacy download en_core_web_sm
   ```

## How to Use

**Start the Streamlit application**:
```bash
streamlit run app.py
```

The app will open automatically in your browser (or navigate to `http://localhost:8501`)

### Using the Web Interface

1. **Enter Review**: Type or paste a movie review in the text area (max 2000 characters)
2. **View Statistics**: See word and character count update in real-time
3. **Try Examples**: Click "Positive", "Negative", or "Mixed" buttons to load example reviews
4. **Analyze**: Click "Analyze Sentiment" button to get predictions
5. **View Results**:
   - Sentiment label (Positive/Negative) with color-coded message
   - Confidence percentage with visual progress bar
   - Interpretive message based on confidence level:
     - **≥90%**: "very confident"
     - **≥75%**: "leans toward"
     - **<75%**: "uncertain"
   - Expand "See full details" for raw metrics

## How It Works

### Text Preprocessing Pipeline
1. Convert text to lowercase
2. Tokenize using SpaCy (with parser and NER disabled for speed)
3. Remove punctuation and whitespace tokens
4. Apply lemmatization (e.g., "running" → "run")
5. Map tokens to vocabulary indices
6. Pad or truncate to 200 tokens
7. Convert to PyTorch tensor

### Inference Pipeline
1. Preprocess input review
2. Embed tokens using GloVe (15,000 → 100-dimensional)
3. Apply embedding dropout (0.4)
4. Process through 2-layer BiLSTM (forward + backward)
5. Concatenate final hidden states from both directions (256-dimensional)
6. Apply output dropout (0.6)
7. Pass through linear layer
8. Apply sigmoid activation to get probability (0-1)
9. Classify: Positive if prob ≥ 0.5, else Negative
10. Calculate confidence as distance from decision boundary (0.5)

## Dependencies

- **streamlit** (1.28.2): Web interface framework
- **torch** (2.2.2): Deep learning framework
- **spacy** (3.7.4): NLP and text preprocessing
- **numpy** (1.26.4): Numerical computing
- **datasets** (2.14.6): Dataset utilities
- **ipykernel** (6.25.0): Jupyter support
- **en_core_web_sm** (3.7.1): SpaCy English model

See `requirements.txt` for complete list with pinned versions.

## Training the Model

To retrain or understand the training process:

1. Navigate to training directory
   ```bash
   cd training
   ```

2. Open the Jupyter notebook
   ```bash
   jupyter notebook Training.ipynb
   ```

3. Follow the notebook cells which:
   - Download IMDb dataset (25,000 reviews)
   - Build vocabulary from training data
   - Load and initialize GloVe embeddings
   - Create BiLSTM model architecture
   - Train with frozen embeddings first
   - Fine-tune with unfrozen embeddings
   - Evaluate on test set
   - Save model, config, and vocabulary

Training includes:
- Early stopping to prevent overfitting
- Learning rate scheduling
- Checkpoint saving
- Validation monitoring

**Note**: Training requires significant compute. GPU recommended via `torch.device("cuda")`.

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'streamlit'"**
- Solution: Install dependencies: `pip install -r requirements.txt`

**"OSError: [E050] Can't find model 'en_core_web_sm'"**
- Solution: Download the model: `python -m spacy download en_core_web_sm`

**"FileNotFoundError: training/models/sentiment_model.pt"**
- Solution: Ensure running from project root directory or check model files exist

**Streamlit not opening in browser**
- Solution: Manually navigate to `http://localhost:8501`

**"Address already in use" on port 8501**
- Solution: Use different port: `streamlit run app.py --server.port 8502`

**Slow inference**
- Normal for CPU (<100ms). For better performance, modify code to use GPU

## Project Statistics

- **Training/Test Split**: Standard IMDb split
- **Vocabulary**: 15,000 most frequent words
- **Model Size**: ~2.5MB (weights only)
- **Dataset**: 25,000 movie reviews
- **Avg Review Length**: ~230 tokens

## Example Usage

### Positive Review
"This movie was amazing! The acting was superb and the plot kept me engaged throughout. Definitely worth watching!"
- **Expected**: Positive (high confidence)

### Negative Review
"Terrible film. Boring plot, bad acting, waste of time."
- **Expected**: Negative (high confidence)

### Mixed Review
"Good production values but the story was boring. Some scenes were excellent, others dragged on."
- **Expected**: May be uncertain due to mixed signals

## Future Enhancements

- Fine-grained ratings (1-5 stars instead of binary)
- Aspect-based sentiment (identify which aspects are positive/negative)
- Attention visualization for model interpretability
- Model quantization for faster inference
- Multi-language support
- Batch prediction capability
- Alternative datasets (Amazon, Twitter, etc.)

## License

Educational and research purposes.

## Acknowledgments

- **IMDb Dataset**: Large Movie Review Dataset by Maas et al.
- **GloVe Embeddings**: Stanford NLP Group
- **Libraries**: PyTorch, Streamlit, SpaCy
