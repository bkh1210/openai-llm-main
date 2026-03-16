def load_prompt(path="prompts/presentation.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()