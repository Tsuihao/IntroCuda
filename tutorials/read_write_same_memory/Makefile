NVCC=nvcc


CUDA_INCLUDEPATH=/usr/local/cuda/include

######################################################
# On Macs the default install locations are below    #
# ####################################################


NVCC_OPTS=-O3 -arch=sm_70 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

student: read_write_same_memory.o Makefile
	$(NVCC) -o read_write -L $(NVCC_OPTS)

main.o: read_write_same_memory.cu gputimer.h
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

read_write_same_memory.o:
	nvcc -c read_write_same_memory.cu $(NVCC_OPTS)

clean:
	rm -f *.o
