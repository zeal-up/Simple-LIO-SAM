#!/bin/bash

declare script_dir=$(cd $(dirname $0); pwd -P)
declare docker_image_tag="v1.0"
declare docker_image_name="spl_lio_sam"
declare docker_image="${docker_image_name}:${docker_image_tag}"
declare container_name="spl_lio_sam"

if [[ $(docker ps -a | grep ${container_name}) ]]; then
    echo "The container '${container_name}' is already runable, start or exec it"
    exit 1
fi

if [[ $(docker images | grep ${docker_image_name} | grep ${docker_image_tag}) ]]; then
    echo "The image has already exists, no need to pull"
else
    docker pull ${docker_image}
fi

docker run -it \
    -v $HOME:/home/splsam\
    -u $(id -u):$(id -g) \
    --network=host \
    -e HOME=/home/splsam \
    --name=${container_name} \
    -e "QT_X11_NO_MITSHM=1" \
    -e DISPLAY=unix$DISPLAY \
    --workdir=/home/splsam \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env="DISPLAY" \
    --privileged \
    -d ${docker_image}

${script_dir}/docker_into.sh ${container_name}