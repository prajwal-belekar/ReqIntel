import requests
import json

prompt = """Analyze this requirement:
The system should lock a user account after five failed password attempts.

Provide a structured response in JSON format with these fields:
- risk_level: string (must be "Low", "Medium", or "High")
- risk_factors: array of strings
- completeness_score: number between 0.0 and 1.0
- missing_requirements: array of strings
- recommendations: array of strings
- explanation: string

Do NOT include any text before or after the JSON. Do NOT use markdown formatting. Output ONLY the JSON."""

payload = {
    "model": "qwen3:8b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 1024
    }
}
print("Testing with complex prompt...")
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
    print("Status:", r.status_code)
    if r.status_code == 200:
        result = r.json()
        print("Response length:", len(result.get("response", "")))
        print("Thinking length:", len(result.get("thinking", "")))
        print("Response preview:", repr(result.get("response", "")[:100]))
        print("Thinking preview:", repr(result.get("thinking", "")[:100]))
        
        # Try to parse response as JSON
        if result.get("response"):
            try:
                parsed = json.loads(result["response"])
                print("✅ Valid JSON parsed!")
                print("Risk level:", parsed.get("risk_level"))
                print("Completeness:", parsed.get("completeness_score"))
            except json.JSONDecodeError as e:
                print("❌ JSON parse error:", str(e)[:100])
        else:
            print("Response is empty")
            if result.get("thinking"):
                # Try to extract JSON from thinking
                import re
                json_start = result["thinking"].find("{")
                json_end = result["thinking"].rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result["thinking"][json_start:json_end]
                    try:
                        parsed = json.loads(json_str)
                        print("✅ Valid JSON extracted from thinking!")
                        print("Risk level:", parsed.get("risk_level"))
                        print("Completeness:", parsed.get("completeness_score"))
                    except json.JSONDecodeError as e:
                        print("❌ Extracted JSON parse error:", str(e)[:100])
    else:
        print("Error:", r.text)
except Exception as e:
    print("Exception:", e)