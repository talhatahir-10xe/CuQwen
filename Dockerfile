FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::http::Timeout="30" \
           -o Acquire::Retries="3" \
           update \
 && apt-get install -y --no-install-recommends \
     build-essential \
     cmake \
     pkg-config \
     libpcre2-dev \
     nlohmann-json3-dev \
     python3-pip \
     python3-dev \
     ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    huggingface_hub \
    numpy

WORKDIR /workspace