.PHONY: clean all

CC=nvcc
CFLAGS= 
LINK_CUDA=-I ${CUDA_TOOLKIT}/include -ldir ${CUDA_TOOLKIT}/nvvm/libdevice/ -L ${CUDA_TOOLKIT}/lib -L ${cudatoolkit.lib}/lib  --dont-use-profile -G -rdc=true -lcudadevrt -lcublas -lcudart


SOURCES = main.cu matrix.cpp matmul.cu runner.cu
HEADERS = matrix.h matmul.h runner.h

all: main

main: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $@  $(LINK_CUDA)
	patchelf --set-rpath "/run/opengl-driver/lib:"$$(patchelf --print-rpath main) main
