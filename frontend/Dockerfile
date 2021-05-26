FROM python:slim
LABEL maintainer="txnguyen2921"
LABEL description="frontend for cat_vs_dog app"

WORKDIR /home/txnguyen2921

COPY . .

RUN echo "Installing dependencies..." && \
    pip3 install -r requirements.txt && \
    echo "Done installing!" && \
    echo "Test environment.." && \
    python -c "import dash" && \
    echo "Environment installed successfully!"

CMD ["python3", "app.py"]

