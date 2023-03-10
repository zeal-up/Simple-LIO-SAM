#!/bin/bash

CONTAINER=$1

xhost +local:root

docker exec -it \
    -u $(id -u):$(id -g) \
    ${CONTAINER:=spl_lio_sam} \
    /bin/bash
