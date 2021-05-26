FROM python:3.8.10-slim
LABEL maintainer="txnguyen2921"
LABEL description="backend for dog_vs_cat app"

WORKDIR /home/txnguyen2921/

COPY . .

RUN apt-get update && \
    apt-get --assume-yes install build-essential && \
    apt-get --assume-yes install wget && \
    make download_and_prepare_model && \
    pip install --user virtualenv && \
    python3 -m venv backend && \
    . backend/bin/activate && \
    pip install -r requirements.txt 

ENTRYPOINT ["source", "backend/bin/activate"]
CMD ["python", "main.py"]