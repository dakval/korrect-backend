from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pykospacing import Spacing
from hanspell import spell_checker

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정: 프론트엔드에서의 접근 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 korrect 도메인만 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 데이터 형식
class SentenceInput(BaseModel):
    sentence: str

# 교정 API
@app.post("/api/correct")
def correct_text(item: SentenceInput):
    sentence = item.sentence

    # PyKoSpacing으로 띄어쓰기 먼저 교정
    spacing = Spacing()
    spaced = spacing(sentence)

    # hanspe

