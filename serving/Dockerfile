FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml ./
RUN uv pip install --system fastapi uvicorn[standard] openai pydantic slowapi anthropic python-dotenv
COPY serving/ ./serving/
COPY dataset/schemas.py ./dataset/schemas.py
COPY dataset/__init__.py ./dataset/__init__.py
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
