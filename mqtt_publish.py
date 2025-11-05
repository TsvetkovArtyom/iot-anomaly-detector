import paho.mqtt.client as mqtt
import json
import random
import time

# Создаем экземпляр клиента MQTT
client = mqtt.Client()

# Подключаемся к локальному брокеру MQTT
client.connect("127.0.0.1", 1883, 60)  # подключение к локальному брокеру

# Тема для публикации
topic = "iot/flows/test123"

# Пример нормальных данных
normal_data = {
    "timestamp": "2025-11-05T12:00:00",
    "device_id": "sensor-01",
    "src_ip": "192.168.1.5",
    "dst_ip": "192.168.1.1",
    "src_port": 80,
    "dst_port": 443,
    "protocol": "TCP",
    "bytes": 1200,
    "packets": 5
}

# Пример атакующих данных
attack_data = {
    "timestamp": "2025-11-05T12:30:00",
    "device_id": "camera-07",
    "src_ip": "192.168.1.20",
    "dst_ip": "192.168.1.1",
    "src_port": 55000,
    "dst_port": 80,
    "protocol": "TCP",
    "bytes": 800000,
    "packets": 35000
}

# Функция для отправки нормальных и атакующих данных
def send_data():
    # Отправка нормального сообщения
    print("Sending normal data...")
    client.publish(topic, json.dumps(normal_data))  # Отправляем нормальные данные
    time.sleep(1)  # Ждем 1 секунду

    # Отправка атакующего сообщения
    print("Sending attack data...")
    client.publish(topic, json.dumps(attack_data))  # Отправляем атакующие данные
    time.sleep(1)  # Ждем 1 секунду

# Запуск отправки данных
for _ in range(10):  # Отправим 10 циклов данных (нормальные и атакующие)
    send_data()

# Отключаемся от брокера
client.disconnect()