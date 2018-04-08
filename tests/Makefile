ROOT = $(shell pwd)
BUILD_DIR = $(ROOT)/build/$(shell uname)

ifeq ($(shell uname), Darwin)
	NPROC = $(shell sysctl -n hw.ncpu)
else
	NPROC = $(shell nproc)
endif

tests:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); cmake $(ROOT); make -j $(NPROC)
