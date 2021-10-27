FROM python:3.8-slim
WORKDIR /FastAPI
COPY requirements.txt /FastAPI/requirements.txt
RUN pip install -r requirements.txt

COPY . /FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
