FROM python:3.8.10-slim
LABEL maintainer="txnguyen2921"
LABEL description="backend for dog_vs_cat app"

WORKDIR /home/txnguyen2921/

COPY . .

RUN apt-get update && \
    apt-get --assume-yes install build-essential && \
    apt-get --assume-yes install wget && \
    make download_and_prepare_model && \
    pip install --user virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -r requirements.txt 

CMD ["python", "main.py"]
