ImageBreed DroneImageScripts
===

This repository contains all Python drone image processing scripts which are used by the ImageBreed (http://imagebreed.org) plant breeding and genetics image processing web-database pipeline. The web-database is based on the BreedBase open-source system for plant breeding and genetics (https://github.com/solgenomics/sgn)

To pull in and run a standalone docker image, so that you can use these scripts, use the following command. This docker image has OpenCV and all other dependencies installed (https://hub.docker.com/r/nmorales3142/imagebreeddroneimagescripts).

`sudo docker run -t -d nmorales3142/imagebreeddroneimagescripts:1.02`

To enter running docker container, where all DroneImageScripts are runnable:

`sudo docker exec -it CONTAINERID bash`

Alternatively, to build fresh standalone docker image, so that you can use these scripts, use the following command. This docker image has OpenCV and all other dependencies installed.

`sudo docker build -t imagebreeddroneimagescripts .`

To launch the built docker image:

`sudo docker run -t -d BUILTIMAGEID`

To enter running docker container, where all DroneImageScripts are runnable:

`sudo docker exec -it RUNNINGCONTAINERID bash`

