import requests
import json

# Test without system parameter
print("Test: No system parameter")
payload = {
    "model": "qwen3:8b",
    "prompt": "Say exactly: Hello World",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 10
    }
    # Note: no 'system' parameter
}
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
    print("Status:", r.status_code)
    if r.status_code == 200:
        result = r.json()
        print("Response:", repr(result.get("response", "")))
        print("Thinking length:", len(result.get("thinking", "")))
        print("All keys:", list(result.keys()))
    else:
        print("Error:", r.text)
except Exception as e:
    print("Exception:", e)