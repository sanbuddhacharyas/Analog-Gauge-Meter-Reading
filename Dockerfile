FROM python:3.9

WORKDIR /usr/src/app

COPY . .

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "./api.py"]