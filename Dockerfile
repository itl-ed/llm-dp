FROM ubuntu:xenial

# Script based on Guillem Frances aigupf/starter-kit image
# Install required packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    build-essential \
    ca-certificates \
    xutils-dev \
    scons \
    flex \
    bison \
    libboost-dev \
    libjudy-dev \
    libboost-program-options-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*


# Set up environment variables
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \ 
    CXX=g++ \
    HOME=/root \
    BASE_DIR=/root/projects \
    LAPKT_URL=https://github.com/LAPKT-dev/LAPKT-public/archive/master.tar.gz \
    PATH=$PATH:/root/projects/lapkt/compiled_planners


# Create required directories
RUN mkdir -p $BASE_DIR
WORKDIR $BASE_DIR


#################################
# Install & build the LAPKT toolkit
#################################
RUN curl -SL $LAPKT_URL | tar -xz \
    && mv LAPKT-* lapkt \
    && cd lapkt \	
    && cd external/libff \
    && make clean && make depend && make \
    && cd ../../ && mkdir compiled_planners && cd planners \
    && cd ff-ffparser && scons && cp ff ../../compiled_planners/. && cd ..\
    && cd bfs_f-ffparser && scons && cp bfs_f ../../compiled_planners/.


WORKDIR $BASE_DIR/lapkt/compiled_planners
CMD ["bash"]