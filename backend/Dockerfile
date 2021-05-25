FROM python:3.9.4-slim

WORKDIR /app

RUN apt-get update

COPY requirements.txt .
RUN pip install -r requirements.txt 