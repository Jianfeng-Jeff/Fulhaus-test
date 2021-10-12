FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04 as builder

# Install lower level dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y curl python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

From tensorflow/tensorflow:2.6.0-gpu

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt .
# Install the requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get install -y python3.6 \
    && ln -s /usr/bin/python3.6

COPY --from=builder . /usr/src/app
COPY . /usr/src/app


ENV HOME=/usr/src/app

CMD ["python3", "test_furniture_classifier.py"]
EXPOSE 8080
