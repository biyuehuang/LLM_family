export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:0.2.0-b2
export IMAGE_NAME=bmg-test

# 设置代理环境变量
export http_proxy=http://proxy.com:913
export https_proxy=http://proxy.com:913
export no_proxy=localhost,127.0.0.1

# 停止并删除已有的容器
sudo docker rm -f $IMAGE_NAME

# 启动新的容器，并传递代理环境变量
sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --privileged \
        --name=$IMAGE_NAME \
        -v /home/models/:/llm/models/ \
        --shm-size="16g" \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        --entrypoint /bin/bash \
        $DOCKER_IMAGE
        # --entrypoint /bin/bash \

# 进入容器
docker exec -it $IMAGE_NAME bash
