from openai import OpenAI
from config import OPENAI_API_KEY
from loader import load_file

# OpenAI client 생성
client = OpenAI(api_key=OPENAI_API_KEY)

# 파일 불러오기
script = load_file("scripts/presentation_script.txt")
prompt_template = load_file("prompts/edit_prompt.txt")

# 프롬프트 템플릿에 발표 대본 삽입
prompt = prompt_template.format(script=script)

# LLM 요청
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

# 결과 출력
print(response.choices[0].message.content)