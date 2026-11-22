import urllib.request
import urllib.error

url = 'http://localhost:8501'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        content = resp.read()
        print("Length:", len(content))
        print("First 200 bytes:", content[:200].decode(errors='ignore'))
except urllib.error.HTTPError as e:
    print("HTTPError Code:", e.code)
    print("Error Body:", e.read().decode(errors='ignore'))
except Exception as e:
    print("General Exception:", e)
