import requests
import json

# Test 1: Very simple prompt
print("Test 1: Simple prompt")
payload1 = {
    "model": "qwen3:8b",
    "prompt": "Say exactly: Hello World",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 10
    }
}
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload1, timeout=10)
    print("Status:", r.status_code)
    if r.status_code == 200:
        result = r.json()
        print("Response:", repr(result.get("response", "")))
        print("Thinking length:", len(result.get("thinking", "")))
    else:
        print("Error:", r.text)
except Exception as e:
    print("Exception:", e)

# Test 2: Simple JSON
print("\nTest 2: Simple JSON request")
payload2 = {
    "model": "qwen3:8b",
    "prompt": "Return a JSON object with field 'test' set to 'value'.",
    "stream": False,
    "options": {
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 50
    }
}
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload2, timeout=15)
    print("Status:", r.status_code)
    if r.status_code == 200:
        result = r.json()
        print("Response:", repr(result.get("response", "")))
        print("Thinking length:", len(result.get("thinking", "")))
        if result.get("response"):
            try:
                parsed = json.loads(result["response"])
                print("Parsed JSON:", parsed)
            except:
                print("Not valid JSON")
    else:
        print("Error:", r.text)
except Exception as e:
    print("Exception:", e)