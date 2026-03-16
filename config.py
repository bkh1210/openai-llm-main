import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 폴더 기준으로 .env 경로 설정
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# .env 로드
load_dotenv(dotenv_path=ENV_PATH)

# API KEY 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")