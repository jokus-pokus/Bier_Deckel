# app/Dockerfile

#FROM ubuntu:kinetic
FROM python:3.9.7

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/jokus-pokus/Bier_Deckel .

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install 'protobuf~=3.19.0'

ENTRYPOINT ["streamlit", "run"]
CMD ["test.py"]