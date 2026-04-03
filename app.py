import streamlit as st
import requests
import time

FLASK_API_URL = "http://localhost:5000"
PREDICT_URL   = f"{FLASK_API_URL}/predict"

# ── Page Setup ────────────────────────────────────────────────
st.set_page_config(
    page_title = "Movie Sentiment Analyzer",
    page_icon  = None,
    layout     = "centered"
)

# ── Helper: Check if API is running ───────────────────────────
def check_api_health():
    try:
        response = requests.get(FLASK_API_URL, timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False   # API not running

# ── Helper: Call prediction endpoint ──────────────────────────
def get_prediction(text):
    try:
        response = requests.post(
            PREDICT_URL,
            json    = {"text": text},
            timeout = 10
        )

        if response.status_code == 200:
            return response.json()
        else:
            error = response.json().get("error", "Unknown error")
            return {"error": error}

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Flask API. Is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Flask API took too long."}


# ── UI Layout ─────────────────────────────────────────────────

# Header
st.title("Movie Review Sentiment Analyzer")
st.markdown(
    "Powered by **BiLSTM + GloVe embeddings** — "
    "trained on 25,000 IMDb reviews"
)

# ── API Status Banner ──────────────────────────────────────────
api_online = check_api_health()

if api_online:
    st.success("Flask API is online and ready")
else:
    st.error(
        "Flask API is offline. "
    )

st.divider()

# ── Input Section ──────────────────────────────────────────────
st.subheader("Enter a Movie Review")

review = st.text_area(
    label            = "Movie review input",
    label_visibility = "hidden",
    placeholder      = "e.g. This film was a masterpiece. The acting was superb...",
    height           = 160,
    max_chars        = 2000,
    key              = "review"
)

# Character counter
char_count = len(review)
word_count = len(review.split()) if review.strip() else 0
st.caption(f"{word_count} words · {char_count}/2000 characters")

# ── Example Reviews ────────────────────────────────────────────
st.markdown("**Try an example:**")

def load_example(text):
    st.session_state["review"] = text

examples = {
    "Positive example": (
        "This movie was an absolute masterpiece. "
        "The performances were outstanding and the story kept me "
        "engaged from start to finish. Highly recommended!"
    ),
    "Negative example": (
        "Terrible film. The plot made no sense, "
        "the acting was wooden, and I fell asleep halfway through. "
        "Complete waste of my time and money."
    ),
    "Mixed example": (
        "Standard sci-fi fare. Solid effects, average plot, predictable ending."
        "Entertaining enough without standing out."
    )
}

col1, col2, col3 = st.columns(3)

btns = ["Positive example", "Negative example", "Mixed example"]
cols = [col1, col2, col3]
for c, b in zip(cols, btns):
    with c:
        st.button(b, on_click=load_example, args=(examples[b],))

st.divider()

# ── Predict Button ─────────────────────────────────────────────
predict_btn = st.button(
    "Analyze Sentiment",
    type     = "primary",
    disabled = not api_online,    # disable button if API is offline
    use_container_width = True
)

# ── Prediction Result ──────────────────────────────────────────
if predict_btn:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")

    else:
        # Show spinner while waiting for API response
        with st.spinner("Sending to Flask API..."):
            time.sleep(0.3)                        # small delay for UX
            result = get_prediction(review)

        st.divider()

        # ── Error handling ─────────────────────────────────────
        if "error" in result:
            st.error(f"API Error: {result['error']}")

        # ── Success: display result ────────────────────────────
        else:
            label      = result["label"]
            confidence = result["confidence"]
            prob       = result["probability"]

            # Big result display
            if label == "Positive":
                st.success(f"## {label} Review")
            else:
                st.error(f"## {label} Review")

            # Confidence meter
            st.markdown(f"**Confidence: {confidence}%**")
            st.progress(confidence / 100)

            # Interpretation message
            if confidence >= 90:
                st.markdown(f"The model is **very confident** this is a {label.lower()} review.")
            elif confidence >= 75:
                st.markdown(f"The model **leans toward** {label.lower()}.")
            else:
                st.markdown("The model is **uncertain** — this review has mixed signals.")

            # ── Detailed breakdown ─────────────────────────────
            with st.expander("See full details from Flask API"):
                st.json({
                    "label":       label,
                    "confidence":  f"{confidence}%",
                    "probability": prob,
                    "word_count":  word_count
                })

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(
    "Architecture: BiLSTM (2 layers, bidirectional) · "
    "Embeddings: GloVe 100d (frozen) · "
    "Tokenizer: SpaCy · "
    "Flask API: localhost:5000"
)