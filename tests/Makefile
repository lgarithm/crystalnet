ifeq ($(shell uname), Darwin)
	NPROC = $(shell sysctl -n hw.ncpu)
else
	NPROC = $(shell nproc)
endif

tests:
	mkdir -p build; cd build; cmake ..; make -j $(NPROC)
