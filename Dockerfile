FROM python:3.11-slim

# Designed for 2 vCPU / 8GB RAM
# All simulation is in-memory Python no external services required
# Typical peak memory usage: < 512MB

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python env/data_generator.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
