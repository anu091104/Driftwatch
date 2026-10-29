# Containerised model API.
#   docker build -t driftwatch .
#   docker run -p 8000:8000 driftwatch      -> http://localhost:8000/docs
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn joblib fastapi uvicorn
COPY src ./src
COPY api ./api
RUN python -m src.demo --no-mlflow > /dev/null
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
