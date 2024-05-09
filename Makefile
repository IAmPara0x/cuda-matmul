LINK_CUDA=-I ${CUDA_TOOLKIT}/include -ldir ${CUDA_TOOLKIT}/nvvm/libdevice/ -L ${CUDA_TOOLKIT}/lib -L ${cudatoolkit.lib}/lib  --dont-use-profile -G -rdc=true -lcudadevrt -lcublas -lcudart

all: main

main: ./main.cu ./matrix.cpp ./matrix.h ./matmul.h ./matmul.cu
	nvcc matrix.cpp matmul.cu main.cu -o main  ${LINK_CUDA}
	patchelf --set-rpath "/run/opengl-driver/lib:"$$(patchelf --print-rpath main) main
