# Use the specified base image
FROM mcr.microsoft.com/devcontainers/base:jammy

# Install necessary packages and tools
RUN sudo apt update && \
    sudo apt install -y make cmake pkg-config g++ libgflags-dev libgoogle-glog-dev libboost-system-dev libboost-thread-dev libboost-timer-dev python3-pip libgtest-dev vim && \
    sudo apt clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Specify default command (this can be overridden by "postCreateCommand" in devcontainer.json)
CMD [ "/bin/bash" ]
