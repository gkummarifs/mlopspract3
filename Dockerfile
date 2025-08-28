FROM python:3.9-slim-bookworm@sha256:latest

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]