from transformers import pipeline

# load once only
sentiment_model = pipeline(
    "sentiment-analysis",
    device=-1   # CPU (Render doesn't support GPU)
)

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"]*100,2)
    }

# ---------- NEW FAST BATCH FUNCTION ----------
def analyze_sentiment_batch(texts):

    predictions = sentiment_model(texts)

    results=[]
    for r in predictions:
        results.append({
            "label": r["label"],
            "score": round(r["score"]*100,2)
        })

    return results
