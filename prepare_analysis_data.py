import json
import ast


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def split_audio_json(audio_data):
    transcript = audio_data.get("transcript", {})
    audio_metrics = audio_data.get("audio_metrics", {})

    return {
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


def prepare_inputs(audio_file="audio.json", pose_file="pose.json"):
    audio_data = load_data(audio_file)
    pose_data = load_data(pose_file)

    audio_parts = split_audio_json(audio_data)
    pose_parts = split_pose_json(pose_data)

    full_script = audio_parts["full_script"]
    results = audio_parts["results"]
    audio_metrics = audio_parts["audio_metrics"]
    pose = pose_parts["pose"]

    return full_script, results, audio_metrics, pose