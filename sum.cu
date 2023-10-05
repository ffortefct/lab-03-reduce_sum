#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define THREADS 1024

#define SIZE 1024 * 1024L

// returns current (wall-clock) time in milliseconds, as a double
double wtime(void) {
  struct timeval timecheck;
  gettimeofday(&timecheck, NULL);
  return (double)timecheck.tv_sec * 1000 + (double)timecheck.tv_usec / 1000;
}

// initializes a vector with 0 in even positions and 1 in odd positions
// so the number of 1s will be "n/2"
__global__ void init_vector(int *const v, long n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    v[i] = i % 2;
}

// makes v[0] = sum(v[i]) (0<=i<n)
void reduce_sum(int *const v, long n) {
  for (long i = 0; i < n; i++)
    v[0] = v[0] + v[i];
}

// Get size of the array from command line (size is in millions of items)
// Initialize the vector of the given size and report execution time
//       (both for wall clock time and CPU time)
// Count the number of 1s in the array and report execution time
//       (both for wall clock time and CPU time)
int main(int argc, const char **argv) {
  long size, mem_size;
  if (argc > 1)
    mem_size = (size = SIZE * atol(argv[1])) * sizeof(int);
  else
    mem_size = (size = SIZE) * sizeof(int);

  int *vect = (int *)malloc(mem_size);
  assert(vect != NULL);
  int *vect_cu;

  assert(cudaMalloc(&vect_cu, mem_size) == cudaSuccess);
  assert(cudaMemset(vect_cu, 0, mem_size) == cudaSuccess);

  printf("Initializing vector of size %ld\n", mem_size);
  double t1 = wtime();
  clock_t t2 = clock();
  init_vector<<<(mem_size + THREADS - 1) / THREADS, THREADS>>>(vect_cu, size);
  t2 = clock() - t2;
  t1 = wtime() - t1;
  double tf = ((double)t2 / (double)CLOCKS_PER_SEC) * 1000.0;
  printf("init time:   wall=%f ms    cpu=%f ms\n", t1, tf);

  assert(cudaMemcpy(vect, vect_cu, mem_size, cudaMemcpyDeviceToHost) ==
         cudaSuccess);
  assert(cudaFree(vect_cu) == cudaSuccess);

  printf("\nRunning reduction\n");
  t1 = wtime();
  t2 = clock();
  reduce_sum(vect, size);
  t2 = clock() - t2;
  t1 = wtime() - t1;
  tf = ((double)t2 / (double)CLOCKS_PER_SEC) * 1000.0;
  printf(" ref time:   wall=%f ms    cpu=%f ms\n", t1, tf);

  if (vect[0] == mem_size / 2)
    printf("\n   OK sum: %i\n", vect[0]);
  else
    printf("ERROR sum: expected %li, counter %i\n", mem_size / 2, vect[0]);
}
