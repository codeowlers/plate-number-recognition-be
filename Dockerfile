#
FROM python:3.9
FROM mylamour/tesseract-ocr:opencv

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
