import requests
import sys
from pprint import pprint

port = str(sys.argv[1]) if len(sys.argv) >= 2 else '25004'
query = str(sys.argv[2]) if len(sys.argv) >= 3 else "Apple"

url_wiki = f"http://127.0.0.1:{port}/queries"
query_list = [query]
topk = 3
response = requests.post(url_wiki, json={"queries": query_list, "k": topk})
if response.status_code == 200:
    for results in response.json()['answers']:
        for i, res in enumerate(results):
            print(f"[{i}] {res}")
else:
    print(response.text)
