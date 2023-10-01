#FROM pytorch/pytorch
FROM nvidia/cuda:11.0.3-base-ubuntu20.04 AS BASE
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt -y remove $(apt list --installed | grep python3.10 | awk -F/ '{print $1}')
RUN apt-get update && \
apt-get install -y --no-install-recommends apt-utils \
&& apt-get install -y \
software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update && apt-get install -y --no-install-recommends build-essential \
python3.7 python3.7-dev python3-pip python3.7-distutils screen && \
ln -f -s /usr/bin/pip3 /usr/bin/pip && \
ln -f -s /usr/bin/python3.7 /usr/bin/python3 && \
apt-get upgrade -y && apt-get install -y \
--no-install-recommends  \
xvfb mesa-utils \
freeglut3-dev \
wget curl vim git \
python3-numpy \
sudo \
&& rm -rf /var/lib/apt/lists/* \
&& useradd -m docker && echo "docker:docker" \
| chpasswd && adduser docker sudo

WORKDIR /gym-duckietown
COPY . .
RUN pip install --no-cache-dir -e . && python3 -m pip \
install --no-cache-dir pyglet==1.5.11 gym==0.25.1 \
scikit-image torch torchvision pillow scipy pandas \
stable-baselines3 numpy==1.21.6 'shimmy>=0.2.1'

FROM base AS final
COPY . .
