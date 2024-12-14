FROM python:3.10-alpine

WORKDIR /Var_API

COPY requirements.txt ./
COPY ./src ./src

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]