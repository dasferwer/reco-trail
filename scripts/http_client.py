import json
import urllib.error
import urllib.request


class Client:
    def __init__(self, base="http://localhost:8000"):
        self.base = base
        self.token = None

    def request(self, method, path, body=None, headers=None, expected=200):
        sent = dict(headers or {})
        if self.token:
            sent["Authorization"] = "Bearer " + self.token
        if body is not None:
            sent["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path,
            method=method,
            headers=sent,
            data=json.dumps(body).encode() if body is not None else None,
        )
        try:
            response = urllib.request.urlopen(request, timeout=20)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            value = json.loads(response.read())
            if response.status != expected:
                raise AssertionError((method, path, response.status, value))
            return value, response.headers

    def login(self, email):
        result, _ = self.request(
            "POST", "/auth/login", {"email": email, "password": "RecoTrailDemo123!"}
        )
        self.token = result["access_token"]
