ImageBreed DroneImageScripts
===

This repository contains all Python drone image processing scripts which are used by the ImageBreed (http://imagebreed.org) plant breeding and genetics image processing web-database pipeline. The web-database is based on the BreedBase open-source system for plant breeding and genetics (https://github.com/solgenomics/sgn)

To launch standalone dockerfile, so that you can use this scripts:
`sudo docker run -t -d imagebreeddroneimagescripts:latest`

This docker container has OpenCV and all other dependencies installed for ease of use.

To enter running docker container, where all DroneImageScripts are runnable:
`sudo docker exec -it CONTAINERID bash`

