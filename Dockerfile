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

# install cmake for xrootd build
RUN /opt/conda/bin/conda install rhash -y
RUN /opt/conda/bin/conda install cmake -y
RUN /opt/conda/bin/conda install -c conda-forge backports.lzma -y

# install pytorch
RUN /opt/conda/bin/conda install pytorch -y

# install fastai
RUN /opt/conda/bin/conda install -c pytorch pytorch-cpu
RUN /opt/conda/bin/conda install torchvision
RUN /opt/conda/bin/conda install -c fastai fastai

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

# download xrootd
ENV XVER="4.8.5"
RUN curl -k -L -O "http://xrootd.org/download/v$XVER/xrootd-${XVER}.tar.gz"
RUN tar xfz xrootd-${XVER}.tar.gz
RUN mkdir ${WDIR}/xrootd-${XVER}/build
WORKDIR ${WDIR}/xrootd-${XVER}/build
RUN cmake $WDIR/xrootd-$XVER -DCMAKE_INSTALL_PREFIX=$WDIR/usr -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR:PATH=/opt/conda/include/python3.6m -DPYTHON_LIBRARY:FILEPATH=/opt/conda/lib/libpython3.6m.so
RUN make -j 8
RUN make install
ENV LIBRARY_PATH="${LIBRARY_PATH}:${WDIR}/usr/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${WDIR}/usr/lib"

# install dependencies
WORKDIR ${WDIR}
RUN git clone https://github.com/scikit-hep/uproot.git
WORKDIR ${WDIR}/uproot
# setup python version to proceed, use 2.X for anaconda2 and 3.X for anaconda3 builds
#ENV PVER=2.7
ENV PVER=3.6
ENV PYTHONPATH="${WDIR}/usr/lib/python${PVER}/site-packages:${PYTHONPATH}"
RUN mkdir -p /data/usr/lib/python${PVER}/site-packages
RUN python setup.py install --prefix=${WDIR}/usr

# build tfaas
WORKDIR ${WDIR}
RUN git clone https://github.com/vkuznet/MLaaS4HEP.git
ENV PYTHONPATH="${WDIR}/MLaaS4HEP/src/python:${PYTHONPATH}"
ENV PATH="${WDIR}/MLaaS4HEP/src/python:${PATH}"

# run the service
CMD ["tail",  "-f"]
