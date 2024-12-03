FROM python:3.9-slim-bullseye

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ENV APP=/app

# Set working directory
WORKDIR $APP

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download BERT model
RUN mkdir bert-base-uncased && \
    wget -P bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/config.json && \
    wget -P bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin && \
    wget -P bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json && \
    wget -P bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

# Create model directory
RUN mkdir -p /mnt/models

ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV KO_DATA_PATH="/ko-data"
ENV STORAGE_URI=""
ENV STORAGE_CONFIG="/mnt/models"


# Copy data loader code
COPY src $APP

# Set environment variables
ENV MODEL_FILE=/mnt/models/model.joblib
ENV CONFIG_FILE=/mnt/models/config.json
ENV LABEL_MAPPING=/mnt/models/mapping.json


ENTRYPOINT ["python"]
# Command to run the predictor
CMD ["intent_predictor.py"]