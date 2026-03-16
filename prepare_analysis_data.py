import json
import ast


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 1차: 정상 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2차: Python dict 문자열
    return ast.literal_eval(text)


def split_audio_json(audio_data):
    transcript = audio_data.get("transcript", {})
    audio_metrics = audio_data.get("audio_metrics", {})

    return {
        "language": transcript.get("language"),
        "segments": transcript.get("segments", []),
        "full_script": transcript.get("full_text", ""),
        "results": {
            "success": audio_data.get("success"),
            "transcript": {
                "language": transcript.get("language"),
                "segments": transcript.get("segments", [])
            }
        },
        "audio_metrics": audio_metrics
    }


def split_pose_json(pose_data):
    return {
        "pose": pose_data
    }


def to_json_string(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


def prepare_inputs(audio_file="audio.json", pose_file="pose.json"):
    audio_data = load_data(audio_file)
    pose_data = load_data(pose_file)

    audio_parts = split_audio_json(audio_data)
    pose_parts = split_pose_json(pose_data)

    full_script = audio_parts["full_script"]
    results = to_json_string(audio_parts["results"])
    audio_metrics = to_json_string(audio_parts["audio_metrics"])
    pose = to_json_string(pose_parts["pose"])

    return full_script, results, audio_metrics, pose

if __name__ == "__main__":

    full_script, results, audio_metrics, pose = prepare_inputs("audio.json", "pose.json")

    print("===== FULL SCRIPT =====")
    print(full_script[:500])  # 너무 길어서 500자만 출력

    print("\n===== RESULTS =====")
    print(results[:500])

    print("\n===== AUDIO METRICS =====")
    print(audio_metrics)

    print("\n===== POSE =====")
    print(pose[:500])