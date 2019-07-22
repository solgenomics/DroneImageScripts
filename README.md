ImageBreed DroneImageScripts
===

This repository contains all Python drone image processing scripts which are used by the ImageBreed (http://imagebreed.org) plant breeding and genetics image processing web-database pipeline. The web-database is based on the BreedBase open-source system for plant breeding and genetics (https://github.com/solgenomics/sgn)



To build standalone docker image, so that you can use these scripts, use the following command. This docker image has OpenCV and all other dependencies installed for ease of use, within the Ubuntu 18.04.
`sudo docker build -t imagebreeddroneimagescripts .`

To launch the built docker image:
`sudo docker run -t -d imagebreeddroneimagescripts:latest`

To enter running docker container, where all DroneImageScripts are runnable:
`sudo docker exec -it CONTAINERID bash`

