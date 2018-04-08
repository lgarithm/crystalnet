ROOT = $(shell pwd)
BUILD_DIR = $(ROOT)/build/$(shell uname)

CMAKE_FLAGS = \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_BUILD_TYPE=Release \

main:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(ROOT); make
