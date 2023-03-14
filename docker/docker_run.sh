#!/bin/bash

declare script_dir=$(cd $(dirname $0); pwd -P)

declare ARGS=$(getopt -o h,c:d: --long help,code:,data: -n "$0" -- "$@")
# echo ARGS=[$ARGS]
eval set -- "${ARGS}"
# echo formatted parameters=[$@]

declare docker_image_tag="v1.0"
declare docker_image_name="zeallin/spl_lio_sam"
declare docker_image="${docker_image_name}:${docker_image_tag}"
declare container_name="spl_lio_sam"
declare code_dir=""
declare data_dir=""

function Help() {
cat << EOF
docker_run.sh -- help script to run docker

Usage:
    docker_run.sh [-h|--help] [-c|--code]

    -h|--help           Show help message
    -c|--code           Code directory mount into container
    -d|--data           Data directory mount into container

EOF
}

if [[ $1 ]]; then
    while true;
    do
        case $1 in
            -h|--help):
                Help; exit 0; ;;
            -c|--code):
                code_dir=$2; shift 2; ;;
            -d|--data):
                data_dir=$2; shift 2; ;;
            --)
                shift; break; ;;
            *)
                echo "unrecognized arguments ${@}"
                Help
                exit 1
                ;;
        esac
    done
fi

if [[ $(docker ps -a | grep ${container_name}) ]]; then
    echo "The container '${container_name}' is already runable, start or exec it"
    exit 1
fi

if [[ $(docker images | grep ${docker_image_name} | grep ${docker_image_tag}) ]]; then
    echo "The image has already exists, no need to pull"
else
    docker pull ${docker_image}
fi

declare volumes=""
if [[ ${code_dir} ]]; then
    volumes="-v ${code_dir}:/home/splsam/codes"
fi
if [[ ${data_dir} ]]; then
    volumes="${volumes} -v ${data_dir}:/home/splsam/data"
fi

docker run -it \
    -u $(id -u):$(id -g) \
    --network=host \
    -e HOME=/home/splsam \
    ${volumes} \
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