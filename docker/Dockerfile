FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y build-essential cmake git
RUN apt-get install -y python3-dev
RUN apt-get install -y libgtk2.0-dev libgtk-3-0 libgtk-3-dev libgtkpkg-config libavcodec-dev libavformat-dev libswscale-dev libhdf5-serial-dev libtbb2 libtbb-dev libjpeg-dev linpng-dev libtiff-dev libv41-dev libxvidcore-dev libatlas-base-dev gfortran
RUN apt-get install -y libgdal-dev
RUN apt-get install -y exiftool libzbar-dev

RUN pip install virtualenv virtualenvwrapper
RUN export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
RUN mkvirtualenv -p python3 cv
RUN workon cv
RUN pip install numpy matplotlib pillow statistics PyEfixTool pytz pysolar scikit-image packaging pyzbar

RUN git clone https://github.com/opencv/opencv
RUN git clone https://github.com/opencv/opencv_contrib
RUN cd opencv
RUN mkdir build
RUN cd build

