# NVIDIA显卡
# This is an auto generated Dockerfile for ros:ros-base
# generated from docker_images/create_ros_image.Dockerfile.em
FROM osrf/ros:humble-desktop-full

# ros2 package
RUN apt-get update \
    && apt install -y ros-humble-plotjuggler-ros \
    && apt install -y ros-humble-sensor-msgs-py \
    && apt install -y ros-humble-diagnostic-updater \
    && apt install -y ros-humble-geographic-msgs \ 
    && apt install -y ros-humble-geodesy \
    && apt install -y libgeographic-dev \ 
    && apt install -y ros-humble-perception-pcl \
    && apt install -y ros-humble-pcl-msgs \
    && apt install -y ros-humble-vision-opencv \
    && apt install -y ros-humble-xacro \
    && apt install -y git \
    && apt install -y wget curl \
    && apt install -y vim \
    && apt install -y software-properties-common \
    && add-apt-repository ppa:borglab/gtsam-develop \
    && apt update \
    && apt install -y libgtsam-dev libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*


RUN useradd --create-home --no-log-init --shell /bin/bash splsam \
&& adduser splsam sudo \
&& usermod -a -G sudo splsam \
&& echo 'splsam:splsam' | chpasswd \
&& mkdir -p /etc/sudoers.d \
&& echo "splsam ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/splsam \
&& chmod 0440 /etc/sudoers.d/splsam \
&& USER=splsam \
&& GROUP=splsam \
&& curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4.1/fixuid-0.4.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - \
&& chown root:root /usr/local/bin/fixuid \
&& chmod 4755 /usr/local/bin/fixuid \
&& mkdir -p /etc/fixuid \
&& printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml \
&& mkdir -p /home/splsam/ros_ws \
&& sudo chown splsam:splsam /home/splsam/ros_ws \
&& echo 'source /opt/ros/humble/setup.bash' >> /home/splsam/.bashrc

# fix liosam 4.1.0
COPY ./source/concepts.h /usr/include/gtsam/base/

USER splsam:splsam
WORKDIR /home/splsam
ENTRYPOINT ["fixuid"]

CMD ["/bin/bash"]

