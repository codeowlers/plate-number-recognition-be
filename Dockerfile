FROM ubuntu:latest
MAINTAINER Developer Sidani "developer.sidani@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip3 python-dev build-essential
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get -y install tesseract-ocr
WORKDIR /code
#
COPY ./requirements.txt /code/requirements.txt
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
