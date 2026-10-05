import requests
import json

prompt = "You are an expert software requirements analyzer. You MUST output ONLY a valid JSON object with exactly these fields: risk_level: string (must be \"Low\", \"Medium\", or \"High\"), risk_factors: array of strings, completeness_score: number between 0.0 and 1.0, missing_requirements: array of strings, recommendations: array of strings, explanation: string. Do NOT include any text before or after the JSON. Do NOT use markdown formatting. Output ONLY the JSON."

system_prompt = "Analyze this requirement: The system should lock a user account after five failed password attempts."

payload = {
    "model": "qwen3:8b",
    "prompt": prompt,
    "system": system_prompt,
    "stream": False,
    "options": {
        "temperature": 0.9,
        "top_p": 0.95,
        "num_predict": 1024
    }
}
print("Testing JSON request with temp=0.9, top_p=0.95...")
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
    print("Status:", r.status_code)
    if r.status_code == 200:
        result = r.json()
        resp_len = len(result.get("response", ""))
        think_len = len(result.get("thinking", ""))
        print("Response length: {}, Thinking length: {}".format(resp_len, think_len))
        if resp_len > 0:
            print("Response preview:", repr(result["response"][:200]))
            # Try to parse as JSON
            try:
                parsed = json.loads(result["response"])
                print("✅ Valid JSON parsed!")
                print("Risk level:", parsed.get("risk_level"))
                print("Completeness:", parsed.get("completeness_score"))
            except Exception as e:
                print("❌ JSON parse error:", str(e)[:100])
        else:
            print("Response is empty or None")
    else:
        print("Error:", r.status_code, r.text[:100])
except Exception as e:
    print("Exception:", e)