ifeq ($(shell uname), Darwin)
	NPROC = $(shell sysctl -n hw.ncpu)
else
	NPROC = $(shell nproc)
endif

# TODO: don't disable assert (remove -DNDEBUG)
CMAKE_FLAGS = \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_BUILD_TYPE=Release \

libmisaka:
	mkdir -p build
	cd build; cmake $(CMAKE_FLAGS) ..; make -j $(NPROC)


release:
# TODO

install: libmisaka
	make -C build install

# TODO: make it work
#run_tests:
#	sh -c "for t in `find build/bin/*_test`; do echo $t; $t; done"

PKG=`cat langs/go/misaka/.goimportpath`
GOPATH=`pwd`/.gopath
# TODO: GOPATH=$(shell mktemp -d)
CGO_CFLAGS="-I`pwd`/src"
CGO_LDFLAGS="-L`pwd`/build/lib"

go: libmisaka
	-rm -fr $(GOPATH)
	mkdir -p $(GOPATH)/src/$(shell dirname $(PKG))
	ln -s `pwd`/langs/go/misaka $(GOPATH)/src/$(PKG)
	CGO_CFLAGS=$(CGO_CFLAGS) CGO_LDFLAGS=$(CGO_LDFLAGS) GOPATH=$(GOPATH) go install -v $(PKG)/example

go_example: go
	LD_LIBRARY_PATH=build/lib ./.gopath/bin/example

python_example: libmisaka
	./examples/python/mnist_slp.py

image:
	docker build .

alpine_image:
	docker build -f Dockerfile.alpine .

lint:
	cpplint src/misaka/*.h

pylint:
	pylint langs/python/misaka.py

format:
	clang-format -i src/misaka/**/*
	clang-format -i src/misaka/*.hpp
	clang-format -i src/*.h
	clang-format -i examples/c/*
	clang-format -i examples/cpp/*
	yapf -i langs/python/misaka.py

tidy:
	./utils/tidy.sh

test: libmisaka
	make -C tests
	./utils/test.sh

check: libmisaka
	make -C tests
	./utils/check-leak.sh

clean:
	-rm -fr build
