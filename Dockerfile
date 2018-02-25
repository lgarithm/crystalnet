FROM ubuntu:bionic

RUN apt update && \
    apt install -y cmake g++ python3 golang-go valgrind cloc
COPY . /misaka
WORKDIR /misaka
RUN make install && \
    make test && \
    make python_example && \
    make go_example && \
    make check && \
    cloc src/misaka
