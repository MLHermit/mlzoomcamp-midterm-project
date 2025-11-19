import requests

link = 'http://127.0.0.1:8000/predict'
client = {'timestamp': '2025-09-01 15:16:08', 'lat': 7.681686, 'lon': 12.088482, 'severity': 'severe', 'cause': 'mechanical', 'vehicles_involved': 3, 'injuries': 4}
response = requests.post(link, json= client)
response.status_code
response.json()











