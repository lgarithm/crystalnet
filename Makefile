ROOT = $(shell pwd)
BUILD_DIR = $(ROOT)/build/$(shell uname)

ifeq ($(shell uname), Darwin)
	NPROC = $(shell sysctl -n hw.ncpu)
else
	NPROC = $(shell nproc)
endif

default: libcrystalnet _tests

CMAKE_FLAGS = \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_BUILD_TYPE=Release \

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(ROOT)

libcrystalnet: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)

release:
# TODO

install: libcrystalnet
	make -C $(BUILD_DIR) install

doc: cmake_targets
	make -C $(BUILD_DIR) doxygen

# TODO: make it work
#run_tests:
#	sh -c "for t in `find build/bin/*_test`; do echo $t; $t; done"

PKG=$(shell cat langs/go/crystalnet/.goimportpath)
GOPATH=$(ROOT)/.gopath
# TODO: GOPATH=$(shell mktemp -d)
CGO_CFLAGS="-I$(ROOT)/src"
CGO_LDFLAGS="-L$(BUILD_DIR)/lib"

go: libcrystalnet
	-rm -fr $(GOPATH)
	mkdir -p $(GOPATH)/src/$(shell dirname $(PKG))
	ln -s $(ROOT)/langs/go/crystalnet $(GOPATH)/src/$(PKG)
	CGO_CFLAGS=$(CGO_CFLAGS) CGO_LDFLAGS=$(CGO_LDFLAGS) GOPATH=$(GOPATH) go install -v $(PKG)/example

go_example: go
	LD_LIBRARY_PATH=$(BUILD_DIR)/lib $(GOPATH)/bin/example

python_example: libcrystalnet
	./examples/python/mnist_slp.py

image:
	docker build .

alpine_image:
	docker build -f Dockerfile.alpine .

lint:
	cpplint src/crystalnet/*.h

pylint:
	pylint langs/python/crystalnet.py

format:
	clang-format -i src/crystalnet/**/*
	clang-format -i src/crystalnet/*.hpp
	clang-format -i src/*.h
	clang-format -i examples/c/*
	clang-format -i examples/cpp/*
	clang-format -i tests/src/crystalnet/**/*
	yapf -i langs/python/crystalnet.py

tidy:
	./utils/tidy.sh

_tests:
	make -C tests

test: libcrystalnet _tests
	./tests/test-includes.sh
	./utils/test.sh

check: libcrystalnet _tests
	./utils/check-leak.sh

clean:
	-rm -fr build
