FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y build-essential cmake git apt-utils
RUN apt-get install -y python3-dev python-pip python3-pip python-numpy
RUN apt-get install -y libgtk2.0-dev libgtk-3-0 libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libhdf5-serial-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libxvidcore-dev libatlas-base-dev gfortran
RUN apt-get install -y libgdal-dev
RUN apt-get install -y exiftool libzbar-dev

RUN pip3 install imutils numpy matplotlib pillow statistics PyExifTool pytz pysolar scikit-image packaging pyzbar \
    && git clone https://github.com/opencv/opencv \
    && git clone https://github.com/opencv/opencv_contrib \
    && cd /opencv_contrib \
    && git checkout 4.1.0 \
    && cd /opencv \
    && git checkout 4.1.0 \
    && mkdir build \
    && cd /opencv/build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
        -D BUILD_EXAMPLES=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_GENERATE_PKGCONFIG=YES .. \
    && make -j 40 \
    && make install \
    && ldconfig \
    && mv /usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/python3.6/dist-packages/cv2/python-3.6/cv2.so

RUN git clone https://github.com/solgenomics/DroneImageScripts \
    && g++ /DroneImageScripts/cpp/stitching_multi.cpp -o /usr/bin/stitching_multi `pkg-config opencv4 --cflags --libs` \
    && g++ /DroneImageScripts/cpp/stitching_single.cpp -o /usr/bin/stitching_single `pkg-config opencv4 --cflags --libs`

RUN apt-get install -y screen htop nano vim
