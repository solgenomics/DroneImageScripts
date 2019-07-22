FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y build-essential cmake git apt-utils
RUN apt-get install -y python3-dev python-pip
RUN apt-get install -y libgtk2.0-dev libgtk-3-0 libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libhdf5-serial-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libxvidcore-dev libatlas-base-dev gfortran
RUN apt-get install -y libgdal-dev
RUN apt-get install -y exiftool libzbar-dev

RUN pip install virtualenv virtualenvwrapper
RUN sed -i '$ a\export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' /root/.bashrc \
    && sed -i '$ a\export WORKON_HOME=/root/.virtualenvs' /root/.bashrc \
    && /bin/bash -c "source /root/.bashrc; source /usr/local/bin/virtualenvwrapper.sh; mkvirtualenv -p python3 cv" \
    && pip install numpy matplotlib pillow statistics PyExifTool pytz pysolar scikit-image packaging pyzbar

RUN git clone https://github.com/opencv/opencv
RUN git clone https://github.com/opencv/opencv_contrib
RUN cd /opencv_contrib \
    && git checkout 4.1.0
RUN cd /opencv \
    && git checkout 4.1.0 \
    && mkdir build \
    && cd /opencv/build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -D PYTHON_EXECUTABLE=/root/.virtualenvs/cv/bin/python \
        -D BUILD_EXAMPLES=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=YES .. \
    && make \
    && make install \
    && ldconfig

RUN git clone https://github.com/solgenomics/DroneImageScripts
RUN g++ /DroneImageScripts/cpp/stitching_multi.cpp -o /usr/bin/stitching_multi `pkg-config opencv4 --cflags --libs`
RUN g++ /DroneImageScripts/cpp/stitching_single.cpp -o /usr/bin/stitching_single `pkg-config opencv4 --cflags --libs`
