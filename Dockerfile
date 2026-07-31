FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000

CMD ["waitress-serve", "--listen=0.0.0.0:8000", "wsgi:app"]
