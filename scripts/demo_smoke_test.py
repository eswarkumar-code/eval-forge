import json
import urllib.request


def get(path):
    with urllib.request.urlopen(f"http://localhost:8000{path}") as r:
        return json.load(r)


def post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(f"http://localhost:8000{path}", data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req) as r:
        return json.load(r)


pending = get("/candidates?status=pending")["candidates"]
groundable = [c for c in pending if c["proposed_expected_output"]]
print(f"approving {len(groundable)} groundable candidates out of {len(pending)} pending")
for c in groundable:
    post(f"/candidates/{c['candidate_id']}/approve")

version = post("/dataset/versions")
print("version:", json.dumps(version))

regression = post("/regression/run")
print("regression:", json.dumps(regression))

metrics = get("/dashboard/metrics")
print("metrics:", json.dumps(metrics))
