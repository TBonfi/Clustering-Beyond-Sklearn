FROM continuumio/anaconda3:2019.10

RUN wget --no-check-certificate -O /tmp/mklml.tgz https://github.com/intel/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz && \
    tar -zxvf /tmp/mklml.tgz && \
    cp -rf mklml_*/* /usr/local/ && \
    rm -rf mklml_*

RUN conda install faiss-cpu -c pytorch