FROM python:3.11-slim

# Zapobieganie buforowaniu wyjścia konsoli i generowaniu plików .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalacja pakietów Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie kodu i danych wejściowych
COPY matches_history.csv .
COPY next_fixtures.csv .
COPY predict_next_gameweek.py .

# Utworzenie folderu na artefakty i bezpiecznego użytkownika
RUN mkdir -p /app/artifacts && useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Domyślne polecenie kontenera
ENTRYPOINT ["python", "predict_next_gameweek.py"]
CMD ["--history", "matches_history.csv", "--fixtures", "next_fixtures.csv", "--output", "artifacts"]