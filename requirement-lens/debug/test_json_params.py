import requests
import json

prompt = '''You are an expert software requirements analyzer. You MUST output ONLY a valid JSON object with exactly these fields:
- risk_level: string (must be "Low", "Medium", or "High")
- risk_factors: array of strings
- completeness_score: number between 0.0 and 1.0
- missing_requirements: array of strings
- recommendations: array of strings
- explanation: string

Do NOT include any text before or after the JSON. Do NOT use markdown formatting. Output ONLY the JSON.'''

system_prompt = '''Analyze this requirement:
The system should lock a user account after five failed password attempts.'''

# Try with different parameters
test_cases = [
    {'temperature': 0.0, 'top_p': 1.0, 'desc': 'Deterministic'},
    {'temperature': 0.3, 'top_p': 0.9, 'desc': 'Low temp'},
    {'temperature': 0.7, 'top_p': 0.9, 'desc': 'Balanced'},
    {'temperature': 0.9, 'top_p': 0.95, 'desc': 'Creative'},
]

base_payload = {
    'model': 'qwen3:8b',
    'prompt': prompt,
    'system': system_prompt,
    'stream': False,
}

for i, params in enumerate(test_cases):
    payload = base_payload.copy()
    payload.update(params)
    print('\nTest {}: {} (temp={}, top_p={})'.format(i+1, params['desc'], params['temperature'], params['top_p']))
    try:
        r = requests.post('http://localhost:11434/api/generate', json=payload, timeout=60)
        if r.status_code == 200:
            result = r.json()
            resp_len = len(result.get('response', ''))
            think_len = len(result.get('thinking', ''))
            print('  Response length: {}, Thinking length: {}'.format(resp_len, think_len))
            if resp_len > 0:
                resp_preview = result['response'][:100]
                print('  Response preview: {}'.format(repr(resp_preview)))
                # Try to parse as JSON
                try:
                    parsed = json.loads(result['response'])
                    print('  ✅ Valid JSON parsed!')
                    print('  Risk level: {}'.format(parsed.get('risk_level')))
                except json.JSONDecodeError as e:
                    print('  ❌ JSON parse error: {}'.format(str(e)[:50]))
            else:
                print('  Response is empty')
        else:
            print('  Error: {}'.format(r.status_code))
    except Exception as e:
        print('  Exception: {}'.format(e))