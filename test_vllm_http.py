import httpx
import json

url = "https://w0wqtv67-8000.usw3.devtunnels.ms/v1/models"
headers = {"Authorization": "Bearer myhpcvllmqwen123"}

try:
    with httpx.Client() as client:
        response = client.get(url, headers=headers)
        print("Status:", response.status_code)
        print("Headers:", dict(response.headers))
        print("Body:", response.text[:500])
except Exception as e:
    print("Error:", e)
