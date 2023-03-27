# Dockerfile

# pull docker image
FROM python:3.8-slim

# set work directory
WORKDIR /ems-checking

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy project
COPY . .