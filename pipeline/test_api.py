import requests

response = requests.post("http://localhost:8000/predict", json={
    "query": "coca cola zero 1л"
})
print("Статус код:", response.status_code)
print("Ответ:", response.json())