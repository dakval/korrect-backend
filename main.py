from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pickle
import re

# 1. 초기화
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 모델 & 데이터 로딩
API_KEY = "AIzaSyCYS9VC2mMLCf8C_TQuJ95JwAedxaBuFpk"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

with open("rag_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embeddings = np.load("rag_embeddings.npy")
embeddings_tensor = torch.tensor(embeddings)

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 3. 요청 바디
class SentenceInput(BaseModel):
    sentence: str

# 4. 맞춤법 교정 API
@app.post("/api/correct")
def correct_text(item: SentenceInput):
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
            f"{i+1}. 문장: {sentence}\n"
            f"참고 문단:\n" + "\n".join(retrieved_chunks)
        )
        prompt_sentences.append(ref_text)

    final_prompt = (
        "다음 문장과 문단을 참고해서 맞춤법을 교정해 주세요. temperature를 고려해서 문맥과 말투를 유지하고, 신조어나 틀리지 않은 단어는 그대로 두세요.\n\n" +
        "\n\n".join(prompt_sentences) +
        "\n\n각 문장에 대해 '문장:'으로 시작하는 줄마다 하나씩 교정해 주세요. 문장의 순서는 번호 순서와 같게 해 주세요. 답변은 번호는 매기지 말고 '문장 :'을 제거하고 쭉 이어서 답해 주세요."
    )

    response = gemini_model.generate_content(
        final_prompt,
        generation_config=genai.types.GenerationConfig(temperature=0.0)
    )

    return {"corrected": response.text}
