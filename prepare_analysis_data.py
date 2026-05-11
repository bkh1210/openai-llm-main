import json
import ast


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def prepare_inputs(
    stt_file="stt.json",
    pitch_file="pitch.json",
    pose_file="pose.json",
    refiner_file="refiner.json"
):
    stt_data = load_data(stt_file)
    pitch_data = load_data(pitch_file)
    pose_data = load_data(pose_file)
    refiner_data = load_data(refiner_file)

    stt = {
        "total_time_sec": stt_data.get("audio_metrics", {}).get("total_time_sec", 0),
        "overall_cpm": stt_data.get("audio_metrics", {}).get("overall_cpm", 0),
        "segments": stt_data.get("audio_metrics", {}).get("segment", []),
        "full_text": stt_data.get("audio_metrics", {}).get("full_text", "")
    }

    pitch = {
        "pitch_metrics": pitch_data.get("pitch_metrics", {}),
        "segments": pitch_data.get("pitch_metrics", {}).get("segment", [])
    }

    posture = {
        "video": pose_data.get("video", {}),
        "timeline": pose_data.get("timeline", [])
    }

    refined = refiner_data.get("refined_result", {})

    return {
        "stt": stt,
        "pitch": pitch,
        "posture": posture,
        "refined": refined
    }