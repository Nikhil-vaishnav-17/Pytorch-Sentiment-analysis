import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def tokenize(text):
    doc = nlp(text.lower())
    return [
        token.lemma_          # "running" → "run"
        for token in doc
        if not token.is_punct  # remove punctuation
        and not token.is_space # remove whitespace
    ]

def encode_and_pad(text, vocab, max_len=200):
    tokens = tokenize(text)

    if len(tokens) >= max_len:
        tokens = tokens[:max_len]   # trim
    else:
        tokens = tokens + ["<PAD>"] * (max_len - len(tokens))  # pad

    return [vocab.get(token, 1) for token in tokens]