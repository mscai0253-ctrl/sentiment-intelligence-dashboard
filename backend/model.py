from transformers import pipeline

# DO NOT load model immediately
sentiment_model = None


def get_model():
    global sentiment_model
    if sentiment_model is None:
        sentiment_model = pipeline(
            "sentiment-analysis",
            device=-1  # CPU
        )
    return sentiment_model


def analyze_sentiment(text):
    model = get_model()
    result = model(text)[0]

    return {
        "label": result["label"],
        "score": round(result["score"] * 100, 2)
    }


def analyze_sentiment_batch(texts):
    model = get_model()
    predictions = model(texts)

    results = []
    for r in predictions:
        results.append({
            "label": r["label"],
            "score": round(r["score"] * 100, 2)
        })

    return results