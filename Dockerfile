FROM python:3.11-slim
WORKDIR /app
COPY ["requirements.txt", "serve.py", "vectorizer_and_model", "./"]
RUN pip install -r requirements.txt
EXPOSE 9696
ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]