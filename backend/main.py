from fastapi import UploadFile
import pandas as pd
import io

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from model import analyze_sentiment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#--------- For Single Text Analysis ---------
@app.post("/analyze")
def analyze(text: str = Form(...)):
    return analyze_sentiment(text)

#--------- For CSV Batch Analysis ---------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile):

    import pandas as pd
    import io

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # ---------- STEP 1 : possible text column names ----------
    possible_names = [
        "review","reviews","text","comment","comments",
        "feedback","message","sentence","content",
        "review_text","customer_review","body","description"
    ]

    text_column = None

    # ---------- STEP 2 : find matching column ----------
    for col in df.columns:
        if col.lower() in possible_names:
            text_column = col
            break

    # ---------- STEP 3 : if not found → choose longest text column ----------
    if text_column is None:
        avg_len = {}
        for col in df.columns:
            try:
                avg_len[col] = df[col].astype(str).str.len().mean()
            except:
                pass

        text_column = max(avg_len, key=avg_len.get)

    # ---------- STEP 4 : analyze ----------
    results=[]
    stats={"POSITIVE":0,"NEGATIVE":0,"NEUTRAL":0}

    for text in df[text_column]:
        result = analyze_sentiment(str(text))
        stats[result["label"]] += 1

        results.append({
            "text":text,
            "label":result["label"],
            "score":result["score"]
        })

    return {
        "detected_column": text_column,
        "total_rows": len(results),
        "stats":stats,
        "results":results
    }

