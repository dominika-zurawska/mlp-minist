FROM python:3.11-slim

WORKDIR /app

ENV MLFLOW_TRACKING_URI=/app/mlflow_data/mlruns

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update && apt-get install -y libgl1

RUN apt-get update && apt-get install -y libglib2.0-0

RUN apt-get update && apt-get install unzip

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY src/ src/
COPY data/ data/

RUN unzip -o data/mnist.zip -d data/

EXPOSE 8501
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY run.sh .
COPY entrypoint.sh .

RUN chmod +x run.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["./run.sh"]
