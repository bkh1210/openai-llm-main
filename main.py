import json
from openai import OpenAI

from config import OPENAI_API_KEY
from loader import load_prompt
from prepare_analysis_data import prepare_inputs


# =========================
# OpenAI Client
# =========================

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 비어 있습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# JSON 응답 정리 함수
# =========================

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

def time_str_to_seconds(time_str: str) -> int:
    minute, second = time_str.split(":")
    return int(minute) * 60 + int(second)


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    minute = seconds // 60
    second = seconds % 60
    return f"{minute}:{second:02d}"


def add_scripts_to_timeline(timeline: list, segments: list) -> list:
    if not isinstance(timeline, list):
        return timeline

    result = []

    for i, item in enumerate(timeline):
        new_item = dict(item)

        item_type = new_item.get("type")
        start_time = new_item.get("time")

        if not start_time:
            result.append(new_item)
            continue

        # start/end는 script 넣지 않음
        if item_type in ["start", "end"]:
            result.append(new_item)
            continue

        start_sec = time_str_to_seconds(start_time)

        if i + 1 < len(timeline):
            next_time = timeline[i + 1].get("time")
            end_sec = time_str_to_seconds(next_time)
        else:
            end_sec = float("inf")

        script_parts = []

        for seg in segments:
            seg_start = int(seg.get("start", 0))

            if start_sec <= seg_start < end_sec:
                text = seg.get("text", "").strip()

                if text:
                    script_parts.append(text)

        new_item["script"] = " ".join(script_parts)

        result.append(new_item)

    return result


def find_segment_time_by_anchor(anchor_text: str, segments: list) -> str | None:
    """anchor_text가 포함된 segment를 찾아 start 시간을 반환한다."""
    if not anchor_text:
        return None

    # 완전 포함 매칭
    for seg in segments:
        if anchor_text in seg.get("text", ""):
            return format_time(seg["start"])

    # 앞 8글자 prefix 매칭 (경미한 불일치 허용)
    prefix = anchor_text[:8] if len(anchor_text) >= 8 else anchor_text
    for seg in segments:
        if prefix in seg.get("text", ""):
            return format_time(seg["start"])

    # 단어 겹침 점수 기반 매칭
    anchor_words = [w for w in anchor_text.split() if len(w) > 1]
    best_seg = None
    best_score = 0
    for seg in segments:
        seg_text = seg.get("text", "")
        score = sum(1 for w in anchor_words if w in seg_text)
        if score > best_score:
            best_score = score
            best_seg = seg

    if best_seg and best_score >= 2:
        return format_time(best_seg["start"])

    return None


def apply_anchor_text_times(timeline: list, segments: list) -> list:
    """timeline의 각 항목에서 anchor_text를 읽어 segment의 실제 start 시간으로 교체한다.
    type이 'end'인 항목은 무조건 마지막 segment의 start 시간으로 설정한다."""
    if not isinstance(timeline, list):
        return timeline

    last_seg_time = format_time(segments[-1]["start"]) if segments else "0:00"

    result = []
    for item in timeline:
        new_item = dict(item)
        anchor_text = new_item.pop("anchor_text", None)

        if new_item.get("type") == "end":
            new_item["time"] = last_seg_time
        elif anchor_text:
            found_time = find_segment_time_by_anchor(anchor_text, segments)
            if found_time:
                new_item["time"] = found_time

        result.append(new_item)

    result.sort(key=lambda x: time_str_to_seconds(x.get("time", "0:00")))

    return result

# =========================
# 사용자 발표 조건 로드
# =========================

def load_user_prompt_json(path: str) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_user_context(user_prompt: dict) -> dict:

    return {
        "topic": user_prompt.get("topic", ""),
        "category": user_prompt.get("category", ""),
        "goal": user_prompt.get("goal", ""),
        "target": user_prompt.get("target", ""),
        "backinfo": user_prompt.get("backinfo", ""),
        "criteria": user_prompt.get("criteria", []),
        "limit_time": user_prompt.get("limit_time", ""),
        "feedback_tone": user_prompt.get("feedback_tone", "DEFAULT")
    }


# 사용자 발표 조건 JSON 로드
user_prompt_data = load_user_prompt_json("dummy_prompt.json")

summary_user_context = build_summary_user_context(
    user_prompt_data
)

# =========================
# 분석 데이터 불러오기
# =========================

analysis_data = prepare_inputs(
    stt_file="stt.json",
    pitch_file="pitch.json",
    pose_file="pose.json",
    refiner_file="refiner.json"
)


# =========================
# 데이터 분리
# =========================

stt = analysis_data["stt"]
pitch = analysis_data["pitch"]
posture = analysis_data["posture"]
refined = analysis_data["refined"]


# =========================
# structure 시간 후보 생성
# =========================

stt_time_candidates = []

for seg in stt["segments"]:

    start = int(seg.get("start", 0))

    m = start // 60
    s = start % 60

    stt_time_candidates.append(f"{m}:{s:02d}")


# 중복 제거
stt_time_candidates = list(dict.fromkeys(stt_time_candidates))


# =========================
# 프롬프트 로드
# =========================

summary_prompt_template = load_prompt("prompts/summary_prompt.txt")


# =========================
# 프롬프트 생성
# =========================

summary_prompt = summary_prompt_template.format(

    # 사용자 발표 조건
    user_context=json.dumps(
        summary_user_context,
        ensure_ascii=False
    ),

    # 전체 분석 데이터
    stt=json.dumps(
        stt,
        ensure_ascii=False
    ),

    pitch=json.dumps(
        pitch,
        ensure_ascii=False
    ),

    posture=json.dumps(
        posture,
        ensure_ascii=False
    ),

    refined=json.dumps(
        refined,
        ensure_ascii=False
    ),

    # 전체 스크립트
    stt_full_text=stt["full_text"],

    # 구조 분석용 segments
    stt_segments=json.dumps(
        stt["segments"],
        ensure_ascii=False
    ),

    # 발표 길이
    stt_total_time_sec=stt["total_time_sec"],

    # structure 시간 제한용
    stt_time_candidates=json.dumps(
        stt_time_candidates,
        ensure_ascii=False
    )
)


# =========================
# GPT 호출
# =========================

summary_response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": summary_prompt
        }
    ],
    temperature=0
)


# =========================
# 응답 처리
# =========================

summary_text = summary_response.choices[0].message.content

summary_json = json.loads(
    clean_json_response(summary_text)
)

if "timeline" in summary_json:
    summary_json["timeline"] = apply_anchor_text_times(
        timeline=summary_json["timeline"],
        segments=stt["segments"]
    )

    summary_json["timeline"] = add_scripts_to_timeline(
        timeline=summary_json["timeline"],
        segments=stt["segments"]
    )

# =========================
# 결과 저장
# =========================

with open(
    "presentation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================
# 출력
# =========================

print(
    json.dumps(
        summary_json,
        ensure_ascii=False,
        indent=2
    )
)

print("presentation_summary.json 저장 완료")