# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentenceInput(BaseModel):
    sentence: str

@app.post("/api/correct")
def correct_text(item: SentenceInput):
    sentence = item.sentence
    # hanspell 제거하고 그대로 반환
    corrected = sentence + " ✅"  # 테스트용 표식
    return {"corrected": corrected}
