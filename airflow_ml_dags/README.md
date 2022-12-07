# MLOPS HW3

### Развертывание Airflow
```
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
```

Путь до модели (model_path) указывается в Airflow Variables как data/models/[date].
