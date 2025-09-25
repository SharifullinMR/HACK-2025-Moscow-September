import requests
import pandas as pd
from datetime import datetime
import os

def process_csv_and_save():
    BASE_URL = "http://localhost:8000"
    
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        print("🩺 Чек живёт сервер или нет:", health.json())
    except:
        print("Сервер не доступен!")
        return

    try:
        with open('submission.csv', 'rb') as f:
            print("Отправка файла")
            
            response = requests.post(
                f"{BASE_URL}/predict-csv",
                files={'file': f},
                timeout=30
            )
        
        print("Статус:", response.status_code)
        
        if response.status_code == 200:
            # Сохраняем полученный CSV напрямую
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"predictions_{timestamp}.csv"
            
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"CSV сохранен: {output_filename}")
            
            df = pd.read_csv(output_filename)
            print(f"Обработано строк: {len(df)}")
            print("Первые 3 строк")
            print(df.head(3))
            
        else:
            print(" Ошибка:", response.text)
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    process_csv_and_save()