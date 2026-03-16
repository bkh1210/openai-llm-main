import json
from prepare_analysis_data import prepare_inputs
from openai import OpenAI
from config import OPENAI_API_KEY
from loader import load_prompt

#prepare_analysis_data에서 가져오기
full_script, results, audio_metrics, pose = prepare_inputs("audio.json", "pose.json")
# OpenAI client 생성
client = OpenAI(api_key=OPENAI_API_KEY)

# 파일 불러오기
prompt_template = load_prompt("prompts/edit_prompt.txt")

# 프롬프트 템플릿에 발표 대본 삽입
prompt = prompt_template.format(
    script=full_script,
    full_script=full_script,
    results=json.dumps(results, ensure_ascii=False),
    audio_metrics=json.dumps(audio_metrics, ensure_ascii=False),
    pose=json.dumps(pose, ensure_ascii=False),
    analysis_mode="summary"
)

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