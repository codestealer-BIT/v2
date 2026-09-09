import sys
import json

for line in sys.stdin:
    data = json.loads(line.strip())
    if "llm_score_v5" in data:
        score = data["llm_score_v5"]
        if score <= 0 or score >= 6:
            print(data["llm_score_v5"])