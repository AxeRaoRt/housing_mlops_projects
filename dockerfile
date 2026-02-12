FROM python:3.12-slim

# Installer uv 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY requirements.txt .

# Installation des de2pendances avec uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

RUN mkdir -p models mlartifacts mlruns

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]