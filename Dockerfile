FROM python:3.13.4

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000
ENV PORT=10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
