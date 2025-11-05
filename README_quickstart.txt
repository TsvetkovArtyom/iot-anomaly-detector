# Шаги для обучения модели и тестирования через MQTT

## 1. Подготовка окружения
### 1.1. Активируем виртуальное окружение
Если виртуальное окружение не активировано, сделай это:

```bash
.venv\Scripts\activate
```

### 1.2. Устанавливаем библиотеки
Убедись, что все нужные библиотеки установлены:

```bash
pip install pandas scikit-learn paho-mqtt
```

## 2. Обучение модели на normal_train_aug.csv
### 2.1. Команда для обучения модели
Теперь, чтобы обучить модель на файле `normal_train_aug.csv`, выполни команду:

```bash
python -m iotids.cli train --input data\normal_train_aug.csv --model artifacts\iso_if.joblib --algo isolation_forest --label-col label --normal-label normal --n-estimators 100 --contamination 0.05
```

### 2.2. Пояснение параметров команды:
- **--input data\normal_train_aug.csv** — путь к файлу с данными для обучения.
- **--model artifacts\iso_if.joblib** — путь для сохранения обученной модели.
- **--algo isolation_forest** — алгоритм для обучения модели.
- **--label-col label** — колонка с метками.
- **--normal-label normal** — метка для нормальных данных.
- **--n-estimators 100** — количество деревьев в модели.
- **--contamination 0.05** — предполагаемая доля аномальных данных.

## 3. Запуск MQTT брокера (Mosquitto)
Если у тебя ещё не установлен Mosquitto, скачай его с [официального сайта](https://mosquitto.org/download/). После установки, запусти его:

```bash
mosquitto -v
```

### 3.1. Проверь, что Mosquitto работает на порту 1883.

## 4. Запуск MQTT воркера для обработки сообщений
Теперь давай запустим **MQTT воркер**, который будет слушать сообщения и классифицировать их с помощью обученной модели.

```bash
python -m iotids.cli mqtt --model artifacts\iso_if.joblib --threshold 0.75 --broker 127.0.0.1 --port 1883 --topic iot/flows/test123
```

### 4.1. Параметры команды:
- **--model artifacts\iso_if.joblib** — путь к обученной модели.
- **--threshold 0.75** — порог классификации (по умолчанию).
- **--broker 127.0.0.1** — локальный брокер.
- **--port 1883** — стандартный порт для MQTT.
- **--topic iot/flows/test123** — топик для публикации и получения сообщений.

## 5. Отправка тестовых данных через MQTT
Для отправки тестовых данных в топик **iot/flows/test123** используем Python скрипт.

### Пример Python скрипта для отправки данных:
```python
import paho.mqtt.client as mqtt
import json
import time

# Создаем экземпляр клиента MQTT
client = mqtt.Client()

# Подключаемся к локальному брокеру MQTT
client.connect("127.0.0.1", 1883, 60)

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

# Отправка данных в топик
print("Sending normal data...")
client.publish(topic, json.dumps(normal_data))  # Отправляем нормальные данные
time.sleep(1)  # Ждем 1 секунду

print("Sending attack data...")
client.publish(topic, json.dumps(attack_data))  # Отправляем атакующие данные
time.sleep(1)  # Ждем 1 секунду

# Отключаемся от брокера
client.disconnect()
```

### 5.1. Запуск скрипта для отправки данных:

```bash
python send_data.py
```

## 6. Проверка выводов
После того как ты запустишь скрипт, и данные будут отправлены в топик, ты сможешь увидеть в консоли MQTT воркера, как система классифицирует эти данные:

```
[INFO] Anomaly detected: False  # Для нормальных данных
[INFO] Anomaly detected: True   # Для атакующих данных
```

---

### Заключение
1. Мы обучили модель на файле `normal_train_aug.csv`.
2. Запустили MQTT брокер (Mosquitto).
3. Настроили MQTT воркер для классификации данных.
4. Отправили тестовые данные через MQTT и проверили результаты.

Теперь система готова для тестирования на новых данных и в реальном времени.

