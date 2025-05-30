from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pickle
import re

# 환경 변수 로딩
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# 서버 인스턴스
app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 로딩
with open("rag_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embeddings = np.load("rag_embeddings.npy")
embeddings_tensor = torch.tensor(embeddings)
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

class SentenceInput(BaseModel):
    sentence: str

@app.post("/api/correct")
def correct_text(item: SentenceInput):
    try:
        input_sentence = item.sentence
        sentences = re.split(r'[!.?]', input_sentence)
        prompt_sentences = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            question_embedding = embedder.encode(sentence, convert_to_tensor=True).cpu()
            cos_scores = util.cos_sim(question_embedding, embeddings_tensor)[0]
            top_results = torch.topk(cos_scores, k=3)
            retrieved_chunks = [chunks[idx] for idx in top_results.indices]

            ref_text = (
                f"{i+1}. 문장: {sentence.strip()}\n"
                f"참고 문단:\n" + "\n".join(retrieved_chunks)
            )
            prompt_sentences.append(ref_text)

        final_prompt = (
            "다음 문장과 문단을 참고해서 맞춤법을 교정해 주세요. temperature를 고려해서 문맥과 말투를 유지하고, 신조어나 틀리지 않은 단어는 그대로 두세요.\n\n"
            + "\n\n".join(prompt_sentences)
            + "\n\n각 문장에 대해 '문장:'으로 시작하는 줄마다 하나씩 교정해 주세요. 문장의 순서는 번호 순서와 같게 해 주세요. 답변은 번호는 매기지 말고 '문장 :
