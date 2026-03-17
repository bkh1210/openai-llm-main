import json
from prepare_analysis_data import prepare_inputs
from openai import OpenAI
from config import OPENAI_API_KEY
from loader import load_prompt

# 1. 데이터 불러오기
full_script, results, audio_metrics, pose = prepare_inputs("audio.json", "pose.json")

# 2. OpenAI client 생성
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 비어 있습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)


# 3. LLM 응답에서 JSON만 추출하는 함수
def clean_json_response(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")

    return text[start:end + 1]


# 4. 중첩 dict 안전하게 꺼내는 함수
def safe_get(d, *keys, default=None):
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


# =========================
# summary prompt 만들기
# =========================
summary_prompt_template = load_prompt("prompts/summary_prompt.txt")

trimmed_script = full_script[:3000]

summary_audio_data = {
    "overall_cpm": safe_get(audio_metrics, "summary", "overall_cpm", default=0),
    "total_time_sec": safe_get(audio_metrics, "summary", "total_time_sec", default=0),
    "speech_time_sec": safe_get(audio_metrics, "summary", "speech_time_sec", default=0),
    "silence_ratio_percent": safe_get(audio_metrics, "summary", "silence_ratio_percent", default=0),
    "filler_count_total": safe_get(audio_metrics, "summary", "filler_count_total", default=0)
}

summary_pose_data = {
    "front_gaze_ratio": pose.get("front_gaze_ratio", 0) if isinstance(pose, dict) else 0,
    "available_pose_keys": list(pose.keys()) if isinstance(pose, dict) else []
}

summary_prompt = summary_prompt_template.format(
    full_script=trimmed_script,
    audio_metrics=json.dumps(summary_audio_data, ensure_ascii=False),
    pose=json.dumps(summary_pose_data, ensure_ascii=False)
)


# =========================
# voice prompt 만들기
# =========================
voice_prompt_template = load_prompt("prompts/voice_prompt.txt")

voice_results_data = {
    "filler_segments": audio_metrics.get("filler_segments", []) if isinstance(audio_metrics, dict) else [],
    "silence_segments": audio_metrics.get("silence_segments", []) if isinstance(audio_metrics, dict) else [],
    "fast_speech_segments": audio_metrics.get("fast_speech_segments", []) if isinstance(audio_metrics, dict) else [],
    "slow_speech_segments": audio_metrics.get("slow_speech_segments", []) if isinstance(audio_metrics, dict) else [],
    "filler_distribution": audio_metrics.get("filler_distribution", {}) if isinstance(audio_metrics, dict) else {}
}

voice_audio_data = {
    "overall_cpm": safe_get(audio_metrics, "summary", "overall_cpm", default=0),
    "filler_count_total": safe_get(audio_metrics, "summary", "filler_count_total", default=0),
    "filler_ratio_percent": safe_get(audio_metrics, "summary", "filler_ratio_percent", default=0),
    "silence_time_sec": safe_get(audio_metrics, "summary", "silence_time_sec", default=0),
    "silence_ratio_percent": safe_get(audio_metrics, "summary", "silence_ratio_percent", default=0)
}

voice_prompt = voice_prompt_template.format(
    results=json.dumps(voice_results_data, ensure_ascii=False),
    audio_metrics=json.dumps(voice_audio_data, ensure_ascii=False)
)


# =========================
# posture prompt 만들기
# =========================
posture_prompt_template = load_prompt("prompts/posture_prompt.txt")

posture_pose_data = pose if isinstance(pose, dict) else {}

posture_prompt = posture_prompt_template.format(
    pose=json.dumps(posture_pose_data, ensure_ascii=False)
)


# =========================
# summary 호출
# =========================
summary_response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": summary_prompt}
    ],
    temperature=0.3
)

summary_text = summary_response.choices[0].message.content
summary_json = json.loads(clean_json_response(summary_text))


# =========================
# voice 호출
# =========================
voice_response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": voice_prompt}
    ],
    temperature=0.3
)

voice_text = voice_response.choices[0].message.content
voice_json = json.loads(clean_json_response(voice_text))


# =========================
# posture 호출
# =========================
posture_response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": posture_prompt}
    ],
    temperature=0.3
)

posture_text = posture_response.choices[0].message.content
posture_json = json.loads(clean_json_response(posture_text))


# =========================
# 최종 결과 합치기
# =========================
final_result = {
    "summary": summary_json,
    "voice": voice_json,
    "posture": posture_json
}


# =========================
# json 파일 저장
# =========================
with open("presentation_analysis.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(json.dumps(final_result, ensure_ascii=False, indent=2))
print("presentation_analysis.json 저장 완료")