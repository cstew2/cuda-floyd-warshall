#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

#include "parallel_floyd_warshall.cuh"

#define min(a,b) (((a)<(b))?(a):(b))
#define BLOCK_SIZE 16

static size_t data_size = 10;

//state variable
static uint64_t rand_state;

__global__ void parallel_floyd_warshall(int *graph, int n, int *path)
{
   // block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * n;
    
    int pathsub = 0;

    for(int a = aBegin, b = bBegin;  a <= aEnd; a += aStep, b += bStep) {
	    //load block into shared memory
	    __shared__ int graph_s[BLOCK_SIZE][BLOCK_SIZE];
	    __shared__ int path_s[BLOCK_SIZE][BLOCK_SIZE];
      	    graph_s[ty][tx] = graph[a + n * ty + tx];
	    path_s[ty][tx] = path[b + n * ty + tx];
	    __syncthreads();
	    
	    //find minimum for block
	    for(int k = 0; k < BLOCK_SIZE; ++k) {
		    pathsub = graph_s[ty][k] < graph_s[ty][k] + path_s[k][tx] ?
			    graph_s[ty][k] : graph_s[ty][k] + path_s[k][tx];
	    }
	    __syncthreads();		   
    }   
    //writeback
    int pathwrite =  n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    path[pathwrite + n * ty + tx] = pathsub;

}

int  *serial_floyd_warshall(int *graph, int n)
{
	int *path = (int *) calloc(sizeof(int), n*n);
	memcpy(graph, path, n*n);
	
	for(int k=0; k < n; k++) {
		for(int i=0; i < n; i++) {
			for(int j=0; j < n; j++) {
				path[(i * n) + j] = min(path[(i * n) + j], path[(i * n) + k]+path[(k * n) + j]);
			}
		}
	}
	return path;
}

struct timespec timer_start()
{
	struct timespec start_time;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
	return start_time;
}

long int timer_end(struct timespec start_time)
{
	struct timespec end_time;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
	long int diff = (end_time.tv_sec - start_time.tv_sec) *
		(long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
	return diff;
}

uint64_t xorshift(void)
{
	uint64_t u = rand_state;
	u ^= u << 12;
	u ^= u << 25;
	u ^= u >> 27;
	rand_state = u;
	return u * 0x2545F4914F6CDD1D;
}

uint64_t xrand(void)
{
	return xorshift(); 
}

void xseed(uint64_t seed)
{
	rand_state = seed;
}

int main(int argc, char ** argv)
{
	//seed RNG
	xseed(time(NULL));
	
	unsigned long int n = 0;
	if(argc != 2) {
		n = data_size;
	}
	else {
		n = atoi(argv[1]);
	}
	
	//create graph
	int *graph = (int *) calloc(sizeof(int), n * n);
	for(int i=0; i < n; i++) {
		for(int j=0; j < n; j++) {
			if(i == j){
				graph[(i * n) + j] = 0;
			}
			else {
				graph[(i * n) + j] = xrand();
			}
		}
	}

	int *path = NULL;
	
	//test serial code speed
	struct timespec start = timer_start();
	path = serial_floyd_warshall(graph, n);
	printf("serial Floyd-Warshall: %li nanoseconds\n", timer_end(start));   

	int *graph_d;
	cudaMalloc(&graph_d, n*n);
	cudaMemcpy(graph_d, graph, n*n, cudaMemcpyHostToDevice);

	int *path_d;
	cudaMalloc(&path_d, n*n);

	int grid = 1;
	int block = 1;
	
	//test parallel code speed
	start = timer_start();
	parallel_floyd_warshall<<<grid, block>>>(graph_d, n, path_d);
	printf("parallel Floyd-Warshall: %li nanoseconds\n", timer_end(start));


	//free memory
	cudaFree(path_d);
	cudaFree(graph_d);
	free(path);
	free(graph);
	return 0;
}
