import requests
import json

try:
    response = requests.post(
        "http://localhost:8030/query",
        data={"query": "what is Field Evaluation Trials FET"},
        timeout=120
    )
    if response.status_code == 200:
        result = response.json()
        ans = result.get("response", "No response")
        print("=" * 80)
        print("RESPONSE:")
        print("=" * 80)
        print(ans)
        print("=" * 80)
    else:
        print(f"Error: {response.status_code}")
        print(response.text[:500])
except Exception as e:
    print(f"Exception: {str(e)}")
