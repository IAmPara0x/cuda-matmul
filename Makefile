.PHONY: clean all

# Compiler and Flags
CUDA_HOME = /opt/cuda
CC = ${CUDA_HOME}/bin/nvcc
CFLAGS = -G -rdc=true -Xcompiler -fPIC --expt-relaxed-constexpr
LINK_FLAGS = -lcudadevrt -lcublas -lcudart
CUDA_INCLUDE_DIR = ${CUDA_HOME}/include
CUDA_LIB_DIR = ${CUDA_HOME}/lib

# Source, Header, and Object Files
SOURCES_CU = main.cu ./src/matmul.cu ./src/runner.cu
SOURCES_CPP = ./src/matrix.cpp
OBJECTS = $(SOURCES_CU:.cu=.o) $(SOURCES_CPP:.cpp=.o)
HEADERS = ./src/matrix.h ./src/matmul.h ./src/runner.h

# Build the Target
all: main

main: $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

# Compile CUDA Sources
%.o: %.cu $(HEADERS)
	$(CC) -c $(CFLAGS) -I$(CUDA_INCLUDE_DIR) $< -o $@

# Compile C++ Sources
%.o: %.cpp $(HEADERS)
	g++ -c -I$(CUDA_INCLUDE_DIR) $< -o $@

# Clean Up
clean:
	rm -f main $(OBJECTS)

