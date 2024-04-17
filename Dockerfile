FROM nvcr.io/nvidia/pytorch:22.04-py3
# install essential softwares
# RUN if [ -e /etc/apt/sources.list.d/cuda.list ] ; then rm /etc/apt/sources.list.d/cuda.list; fi ; 
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt install -y vim zsh git ssh sudo language-pack-en tmux libssl-dev libmysqlclient-dev build-essential
RUN update-locale LANG=en_US.UTF-8
RUN python -m pip install --upgrade pip
RUN python -m pip install reformer-pytorch==1.4.4 wandb h5py opt-einsum

ARG UID 1000
RUN yes | adduser --uid $UID --shell /bin/zsh --disabled-password user -q
RUN ln -s /workspace /home/user/workspace && chown user:user /workspace && chmod 755 /workspace

USER user
WORKDIR /workspace

