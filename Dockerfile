#
FROM python:3.8.2

#
WORKDIR /

#
COPY ./requirements.txt /requirements.txt

#
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

#
COPY . /

#
CMD ["uvicorn", "main:/", "--host", "0.0.0.0", "--port", "8000"]
