FROM python:3.9.5-alpine

WORKDIR /app

ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive


RUN apk add --no-cache bash postgresql-libs postgresql-dev g++ gcc libffi-dev libxslt-dev jpeg-dev zlib-dev linux-headers
RUN pip install --upgrade pip && pip install uwsgi==2.0.19.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python src/manage.py collectstatic --noinput

CMD python src/manage.py migrate && \
    gunicorn vkr.wsgi --chdir /app/src --bind 0.0.0.0 --preload --log-file -