import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 41.8781,
    "longitude": -87.6298,
    "start_date": "2024-06-01",
    "end_date": "2024-06-30",
    "hourly": ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "weather_code",
        "wind_speed_10m",
        "wind_direction_10m"
    ]),
    "timezone": "America/Chicago"
}

response = requests.get(url, params=params)
data = response.json()

weather = pd.DataFrame(data["hourly"])

weather.to_csv("chicago_weather_2024_06_hourly.csv", index=False, encoding="utf-8-sig")

print(weather.head())
print(weather.tail())
print(weather.shape)