# MLOPS HW2

### Сборка Docker-образа
```
docker build -t p0rtale/mlops_hw2 .
```

### Извлечение Docker-образа из DockerHub
```
docker pull p0rtale/mlops_hw2
```

### Запуск контейнера
```
docker run --name mlops_hw2 -p 8000:8000 p0rtale/mlops_hw2
```

### Запуск тестов (в работающем контейнере)
```
docker exec -it mlops_hw2 bash
python3 -m pytest test_predict_request.py
```

### Отправка запросов к сервису
```
# Установка зависимостей: pip3 install -r requirements.txt
python3 make_request.py
```
