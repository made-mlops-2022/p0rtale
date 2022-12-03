FROM python:3.10-slim-buster

WORKDIR /app

COPY server.py test_predict_request.py heart_disease_data.py requirements.txt ./
ADD models app/models

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

ENV PATH_TO_MODEL="app/models/model.pkl"

EXPOSE 8000

CMD python3 server.py
