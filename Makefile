CC=gcc

OPT=0
CFLAGS=-O$(OPT)
SUM_SIZE=2048

sum: sum.c
	$(CC) $(CFLAGS) -o $@ $<

sum-omp: sum.c
	$(CC) $(CFLAGS) -fopenmp -D OMP -o $@ $<

sum-cuda: sum.cu
	nvcc -o $@ $<

sum-pprof: sum.c
	$(CC) $(CFLAGS) -fopenmp -lprofiler -o $@ $<
	CPUPROFILE=main-serial.prof ./$@ $(SUM_SIZE)
	pprof --web ./$@ main-serial.prof

sum-omp-pprof: sum.c
	$(CC) $(CFLAGS) -fopenmp -lprofiler -D OMP -o $@ $<
	CPUPROFILE=main-omp.prof ./$@ $(SUM_SIZE)
	pprof --web ./$@ main-omp.prof

all: sum sum-omp sum-cuda

clean:
	rm -f sum sum-omp sum-cuda sum-pprof sum-omp-pprof main-serial.prof main-omp.prof

.PHONY: clean
