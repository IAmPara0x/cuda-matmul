.PHONY: clean all

# Compiler and Flags
CC = nvcc
CFLAGS = -G -rdc=true -Xcompiler -fPIC
LINK_FLAGS = -lcudadevrt -lcublas -lcudart
CUDA_INCLUDE_DIR = ${CUDA_TOOLKIT}/include
CUDA_LIB_DIR = ${CUDA_TOOLKIT}/lib

# Source, Header, and Object Files
SOURCES_CU = main.cu ./src/matmul.cu ./src/runner.cu
SOURCES_CPP = ./src/matrix.cpp
OBJECTS = $(SOURCES_CU:.cu=.o) $(SOURCES_CPP:.cpp=.o)
HEADERS = ./src/matrix.h ./src/matmul.h ./src/runner.h

# Build the Target
all: main

main: $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ -L$(CUDA_LIB_DIR) $(LINK_FLAGS)
	patchelf --set-rpath "/run/opengl-driver/lib:"$$(patchelf --print-rpath main) main

# Compile CUDA Sources
%.o: %.cu $(HEADERS)
	$(CC) -c $(CFLAGS) -I$(CUDA_INCLUDE_DIR) $< -o $@

# Compile C++ Sources
%.o: %.cpp $(HEADERS)
	g++ -c -I$(CUDA_INCLUDE_DIR) $< -o $@

# Clean Up
clean:
	rm -f main $(OBJECTS)
