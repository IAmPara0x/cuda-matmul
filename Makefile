.PHONY: all build clean

# Compiler and Flags

CMAKE := cmake
BUILD_DIR := build

# Build the Target
all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -C $(BUILD_DIR)

# Clean Up
clean:
	rm -rf ./build
