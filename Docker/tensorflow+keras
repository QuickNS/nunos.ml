# Todo Add flags for training and evaluation

FROM tensorflow/tensorflow:1.7.0-devel-gpu

EXPOSE 6006

RUN apt-get update && apt-get install python-opencv -y && apt-get clean -y
WORKDIR /tensorflow/tensorflow/examples/
RUN git clone https://github.com/fizyr/keras-retinanet.git
RUN cd keras-retinanet && pip install . --user
RUN pip install Pillow h5py opencv-python --user

WORKDIR keras-retinanet

