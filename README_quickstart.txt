# Quickstart — IoT Anomaly Module

## 1) Установка зависимостей
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

## 2) Подготовка файлов
Сохраните модуль как `iot_anomaly_module.py` (скопируйте из чата/канваса).
Положите датасеты из этого пакета: iot_train.csv и iot_test.csv

## 3) Обучение (IsolationForest на нормальном поведении)
python iot_anomaly_module.py train     --input iot_train.csv     --model artifacts/iso_if.joblib     --algo isolation_forest     --label-col label --normal-label normal     --n-estimators 300 --contamination 0.02

## 4) Оценка на тестовом наборе (есть метки)
python iot_anomaly_module.py eval     --input iot_test.csv     --model artifacts/iso_if.joblib     --output scored_test.csv     --label-col label --positive-label attack     --threshold 0.6

Откройте scored_test.csv — там будет колонка anomaly_score.

## 5) Онлайновый REST-сервис
python iot_anomaly_module.py serve     --model artifacts/iso_if.joblib     --threshold 0.6 --host 0.0.0.0 --port 8080

# В другом терминале:
curl -X POST http://127.0.0.1:8080/score   -H "Content-Type: application/json"   -d '{"timestamp":"2025-01-01T12:34:56","device_id":"camera-2","src_ip":"10.0.1.10","dst_ip":"192.168.0.5","src_port":80,"dst_port":1024,"protocol":"TCP","bytes":1200,"packets":5}'

## 6) MQTT-поток (опционально)
# Запустите брокер (например, mosquitto).
python iot_anomaly_module.py mqtt     --model artifacts/iso_if.joblib     --threshold 0.6     --broker localhost --port 1883 --topic iot/flows

# Публикация тестового сообщения:
mosquitto_pub -h localhost -t iot/flows -m '{"timestamp":"2025-01-01T12:45:00","device_id":"thermostat-1","src_ip":"10.0.0.7","dst_ip":"192.168.1.8","src_port":443,"dst_port":8883,"protocol":"TCP","bytes":230000,"packets":12000}'

## Примечания
- Порог 0.6 эмпирический; подбирайте по ROC/PR-кривым или по бизнес-метрикам (Recall/Precision).
- Для автоэнкодера используйте `--algo autoencoder` при train (нужен TensorFlow).
