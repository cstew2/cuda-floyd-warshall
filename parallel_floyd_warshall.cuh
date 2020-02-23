#ifndef __PARALLEL_FLOYD_WARSHALL_H__
#define __PARALLEL_FLOYD_WARSHALL_H__

#include <stdint.h>

//timing functions
struct timespec timer_start();
long int timer_end(struct timespec start_time);

//xorshift* algorithm for PRNG
uint64_t xorshift(void);
uint64_t xrand(void);
void xseed(uint64_t seed);

//floyd-Warshall functions
__global__ void parallel_floyd_warshall(int *graph, int n, int *path);
int *serial_floyd_warshall(int *graph, int n);

#endif