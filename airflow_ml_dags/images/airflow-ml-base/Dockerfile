FROM python:3.8.15-slim-buster

RUN mkdir /build/
COPY requirements.txt /build/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /build/requirements.txt
