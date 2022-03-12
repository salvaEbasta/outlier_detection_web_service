FROM condaforge/mambaforge-pypy3

# work directory
ENV WDIR="/opt/outlier_detection"
ARG WDIR="/opt/outlier_detection"
#ADD . ${WDIR}
COPY . ${WDIR}
WORKDIR ${WDIR}

# conda
RUN apt-get update
#ENV PATH="/root/miniconda3/bin:${PATH}"
#ARG PATH="/root/miniconda3/bin:${PATH}"
#RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
#RUN wget \
#    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#    && mkdir /root/.conda \
#    && bash Miniconda3-latest-Linux-x86_64.sh -b \
#    && rm -f Miniconda3-latest-Linux-x86_64.sh 

#RUN conda config --set restore_free_channel true
COPY ./conda-envs/linux-64.lock .
RUN mamba create --name tesi-env --file linux-64.lock && conda clean -afy
RUN conda activate tesi-env
CMD [ "python", "ml_microservice" ]