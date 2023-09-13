FROM python:3.11

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "fastapi_demo:app", "--host","127.0.0.1", "--port", "8000"]

#CMD python3 -m uvicorn fastapi_demo:app --reload --host 0.0.0.0 --port 8000