import streamlit as st
import torch
import pickle
import json
import time
from training.preprocess import encode_and_pad
from training.SentimentLSTM import SentimentLSTM


# ── Page Setup ────────────────────────────────────────────────
st.set_page_config(
    page_title = "Movie Sentiment Analyzer",
    page_icon  = "🎬",
    layout     = "centered"
)


# ── Load Model Once ───────────────────────────────────────────
# @st.cache_resource ensures model loads only once
# even when Streamlit reruns on every interaction
@st.cache_resource
def load_model():
    # Load vocab dictionary (word → index)
    with open("training/models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load architecture config (vocab_size, embed_dim, hidden_dim)
    with open("training/models/model_config.json", "r") as f:
        config = json.load(f)

    # Rebuild model architecture
    model = SentimentLSTM(
        vocab_size       = config["vocab_size"],
        embed_dim        = config["embed_dim"],
        hidden_dim       = config["hidden_dim"],
        embedding_matrix = torch.zeros(
            config["vocab_size"], config["embed_dim"]
        )  # placeholder — actual weights loaded below
    )

    # Load saved weights
    model.load_state_dict(
        torch.load(
            "training/models/sentiment_model.pt",
            map_location = "cpu"   # always CPU for deployment
        )
    )

    model.eval()   # dropout OFF for inference
    return vocab, model


# ── Inference Function ────────────────────────────────────────
def run_inference(text, vocab, model):
    encoded = encode_and_pad(text, vocab)

    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():                       # no gradients needed
        logit = model(tensor)                   # raw output score
        prob  = torch.sigmoid(logit).item()     # convert to 0-1 probability

    label      = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "label":       label,
        "confidence":  round(confidence * 100, 2),   # e.g. 94.32
        "probability": round(prob, 4),               # e.g. 0.9432
    }


# ── Load model at startup ─────────────────────────────────────
vocab, model = load_model()


# ── UI ────────────────────────────────────────────────────────
st.title("Movie Review Sentiment Analyzer")
st.markdown(
    "Powered by **BiLSTM + GloVe embeddings** — "
    "trained on 25,000 IMDb reviews"
)

st.divider()

# ── Input Section ─────────────────────────────────────────────
st.subheader("Enter a Movie Review")

review = st.text_area(
    label            = "Movie review input",
    label_visibility = "hidden",
    placeholder      = "e.g. This film was a masterpiece. The acting was superb...",
    height           = 160,
    max_chars        = 2000,
    key              = "review"
)

# Word + character counter
char_count = len(review)
word_count = len(review.split()) if review.strip() else 0
st.caption(f"{word_count} words · {char_count}/2000 characters")

# ── Example Buttons ───────────────────────────────────────────
st.markdown("**Try an example:**")

def load_example(text):
    st.session_state["review"] = text

examples = {
    "Positive": (
        "This movie was an absolute masterpiece. "
        "The performances were outstanding and the story kept me "
        "engaged from start to finish. Highly recommended!"
    ),
    "Negative": (
        "Terrible film. The plot made no sense, "
        "the acting was wooden, and I fell asleep halfway through. "
        "Complete waste of my time and money."
    ),
    "Mixed": (
        "Standard sci-fi fare. Solid effects, average plot, predictable ending. "
        "Entertaining enough without standing out."
    )
}

col1, col2, col3 = st.columns(3)
for col, (label, text) in zip([col1, col2, col3], examples.items()):
    with col:
        st.button(label, on_click=load_example, args=(text,))

st.divider()

# ── Analyze Button ────────────────────────────────────────────
predict_btn = st.button(
    "Analyze Sentiment",
    type                = "primary",
    use_container_width = True
)

# ── Result Display ────────────────────────────────────────────
if predict_btn:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")

    else:
        with st.spinner("Analyzing review..."):
            time.sleep(0.3)    # small pause for UX feel
            result = run_inference(review, vocab, model)

        st.divider()

        label      = result["label"]
        confidence = result["confidence"]
        prob       = result["probability"]

        # Sentiment result — green for positive, red for negative
        if label == "Positive":
            st.success(f"## {label} Review")
        else:
            st.error(f"## {label} Review")

        # Confidence bar
        st.markdown(f"**Confidence: {confidence}%**")
        st.progress(confidence / 100)

        # Interpretation message
        if confidence >= 90:
            st.markdown(
                f"The model is **very confident** this is a {label.lower()} review."
            )
        elif confidence >= 75:
            st.markdown(f"The model **leans toward** {label.lower()}.")
        else:
            st.markdown(
                "The model is **uncertain** — this review has mixed signals."
            )

        # Raw details in expander
        with st.expander("See full details"):
            st.json({
                "label":       label,
                "confidence":  f"{confidence}%",
                "probability": prob,
                "word_count":  word_count,
                "char_count":  char_count
            })

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Architecture: BiLSTM (2 layers, bidirectional) · "
    "Embeddings: GloVe 100d (frozen) · "
    "Tokenizer: SpaCy"
)