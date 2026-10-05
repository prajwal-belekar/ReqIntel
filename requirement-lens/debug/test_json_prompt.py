import requests, json
prompt = """
        === REQUIREMENT ===
        The system should lock a user account after five failed password attempts.
        
        === ML CLASSIFICATION ===
        Category: SE
        Confidence: 0.32
        
        === RETRIEVED GUIDELINES ===
        [GUIDELINE 1]
Title: Account Lockout After Failed Attempts
Similarity: 0.75
Text: User accounts should be temporarily locked after multiple consecutive failed login attempts to prevent brute force attacks....

[GUIDELINE 2]
Title: Password Reset Security
Similarity: 0.54
Text: Password reset mechanisms must verify user identity through secure channels before allowing password changes....

[GUIDELINE 3]
Title: Multi-Factor Authentication
Similarity: 0.53
Text: Critical systems should implement multi-factor authentication combining password with additional verification methods....
        """
# System prompt that explicitly requests JSON
system_prompt = 'You are an expert software requirements analyzer. You MUST output ONLY a valid JSON object with exactly these fields: \\n- risk_level: string (must be \"Low\", \"Medium\", or \"High\")  \\n- risk_factors: array of strings\\n- completeness_score: number between 0.0 and 1.0\\n- missing_requirements: array of strings\\n- recommendations: array of strings\\n- explanation: string\\n\\nDo NOT include any text before or after the JSON. Do NOT use markdown formatting. Output ONLY the JSON.'

payload = {
    'model': 'qwen3:8b',
    'prompt': prompt,
    'system': system_prompt,
    'stream': False,
    'options': {'temperature': 0.1, 'top_p': 0.9, 'num_predict': 512}
}
print('Sending request to Ollama with JSON-requesting system prompt...')
r = requests.post('http://localhost:11434/api/generate', json=payload, timeout=120)
print('Status:', r.status_code)
if r.status_code == 200:
    result = r.json()
    print('Response content length:', len(result.get('response', '')))
    print('Response start:', repr(result.get('response', '')[:100]))
    if result.get('response'):
        try:
            parsed = json.loads(result['response'])
            print('✅ Valid JSON parsed successfully!')
            print('Keys:', list(parsed.keys()))
            print('Risk level:', parsed.get('risk_level'))
            print('Completeness:', parsed.get('completeness_score'))
        except json.JSONDecodeError as e:
            print('❌ JSON decode error:', e)
            print('Response:', result['response'][:500])
    else:
        print('Empty response')
        print('Thinking:', result.get('thinking', 'NO THINKING FIELD')[:200])
else:
    print('Error:', r.text)