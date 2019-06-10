FROM continuumio/anaconda3
MAINTAINER Valentin Kuznetsov vkuznet@gmail.com

# add environment
ENV WDIR=/data

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    /opt/conda/bin/conda install jupyter -y && \
    /opt/conda/bin/conda install -c menpo opencv3 -y && \
    /opt/conda/bin/conda install numpy pandas scikit-learn matplotlib pyyaml h5py keras -y && \
    /opt/conda/bin/conda upgrade dask && \
    pip install tensorflow imutils

# install uproot
RUN /opt/conda/bin/conda install -c conda-forge uproot backports.lzma -y

# install pytorch
RUN /opt/conda/bin/conda install pytorch -y

# install fastai
RUN /opt/conda/bin/conda install -c pytorch pytorch-cpu
RUN /opt/conda/bin/conda install torchvision
RUN /opt/conda/bin/conda install -c fastai fastai

# pyarrow for HDFS readers
RUN /opt/conda/bin/conda install -c conda-forge pyarrow -y

# Create new user account
#ENV USER=mlaas
#RUN useradd ${USER} && install -o ${USER} -d ${WDIR}
# add user to sudoers file
#RUN echo "%$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# switch to user
#USER ${USER}

# start the setup
RUN mkdir -p $WDIR/usr
WORKDIR ${WDIR}

# build tfaas
WORKDIR ${WDIR}
RUN git clone https://github.com/vkuznet/MLaaS4HEP.git
ENV PYTHONPATH="${WDIR}/MLaaS4HEP/src/python:${PYTHONPATH}"
ENV PATH="${WDIR}/MLaaS4HEP/bin:${PATH}"

# run the service
CMD ["tail",  "-f"]
