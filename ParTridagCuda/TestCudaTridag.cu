#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h> 
typedef double REAL;
#include "TridagKernel.cu.h"
#include "TridagPar.h"


int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/**
 * solves a segmented tridag, i.e., 
 * solves `n/sgm_size` independent tridag problems.
 * Logically, the arrays should be of size [n/sgm_size][sgm_size],
 * and the segmented tridag corresponds to a map on the outer
 * dimension which is applying tridag to its inner dimension.
 * This is the CUDA parallel implementation, which uses
 * block-level segmented scans. This version assumes that
 * `n` is a multiple of `sgm_sz` and also that `block_size` is
 * a multiple of `sgm_size`, i.e., such that segments do not
 * cross block boundaries.
 */
void tridagCUDAWrapper( const unsigned int block_size,
                        REAL*   a,
                        REAL*   b,
                        REAL*   c,
                        REAL*   r,
                        const unsigned int n,
                        const unsigned int sgm_sz,
                        REAL*   u,
                        REAL*   uu 
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 8 * sizeof(REAL);

    // assumes sgm_sz divides block_size
    if((block_size % sgm_sz)!=0) {
        printf("Invalid segment or block size. Exiting!\n\n!");
        exit(0);
    }
    if((n % sgm_sz)!=0) {
        printf("Invalid total size (not a multiple of segment size). Exiting!\n\n!");
        exit(0);
    }
    num_blocks = (n + (block_size - 1)) / block_size;
    TRIDAG_SOLVER<<< num_blocks, block_size, sh_mem_size >>>(a, b, c, r, n, sgm_sz, u, uu);
    cudaThreadSynchronize();
}

/**
 * solves a segmented tridag, i.e., 
 * solves `n/sgm_size` independent tridag problems.
 * Logically, the arrays should be of size [n/sgm_size][sgm_size],
 * and the segmented tridag corresponds to a map on the outer
 * dimension which is applying tridag to its inner dimension.
 * This is the CPU sequential implementation, but morally the 
 * code is re-written to use (sequential) scans.
 */
void 
goldenSeqTridagPar(
    const REAL*   a,   // size [n]
    const REAL*   b,   // size [n]
    const REAL*   c,   // size [n]
    const REAL*   r,   // size [n]
    const int     n,
    const int     sgm_size,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
) {
    if((n % sgm_size)!=0) {
        printf("Invalid total size (not a multiple of segment size). Exiting!\n\n!");
        exit(0);
    }

    for(int i=0; i<n; i+=sgm_size) {
        tridagPar(a+i, b+i, c+i, r+i, sgm_size, u+i, uu+i);  
    }
}


void init(int block, int n, REAL* a, REAL* b, REAL* c, REAL* d) {
    srand(111);

    // Tridag is numerically unstable if tried with random data,
    // but still ... lets try. We allocate the same data for every block,
    // otherwise we have good chances of hitting a bad case!
    for(int i=0; i<block; i++) {
        a[i] = ((REAL) rand()) / RAND_MAX; 
        b[i] = ((REAL) rand()) / RAND_MAX; 
        c[i] = ((REAL) rand()) / RAND_MAX; 
        d[i] = ((REAL) rand()) / RAND_MAX; 
    }
    for(int i=block; i<n; i++) {
        a[i] = a[i-block];
        b[i] = b[i-block]; 
        c[i] = c[i-block]; 
        d[i] = d[i-block]; 
    }
}

#define N          (1024*1024*8)
#define SGM_SIZE   8
#define BLOCK_SIZE 256
#define EPS        0.002

void validate(int n, REAL* cpu, REAL* gpu) {
    for(int i=0; i<n; i++) {
        REAL div_fact = (fabs(cpu[i]) < 1.0) ? 1.0 : fabs(cpu[i]);
        REAL diff = fabs(cpu[i]-gpu[i])/div_fact;
        if( diff > EPS ) {
            printf("INVALID Result at index %d, %f %f diff: %f. Exiting!\n\n", i, cpu[i], gpu[i], diff);
            exit(0);
        }
    }
}
/*REAL initOperator(  const vector<REAL>& x, 
                    vector<vector<REAL> >& Dxx) {
    const unsigned n = x.size();

    REAL dxl, dxu;

    //  lower boundary
    dxl      =  0.0;
    dxu      =  x[1] - x[0];
    
    Dxx[0][0] =  0.0;
    Dxx[0][1] =  0.0;
    Dxx[0][2] =  0.0;
    Dxx[0][3] =  0.0;
    
    //  standard case
    for(unsigned i=1;i<n-1;i++)
    {
        dxl      = x[i]   - x[i-1];
        dxu      = x[i+1] - x[i];

        Dxx[i][0] =  2.0/dxl/(dxl+dxu);
        Dxx[i][1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[i][2] =  2.0/dxu/(dxl+dxu);
        Dxx[i][3] =  0.0; 
    }

    //  upper boundary
    dxl        =  x[n-1] - x[n-2];
    dxu        =  0.0;

    Dxx[n-1][0] = 0.0;
    Dxx[n-1][1] = 0.0;
    Dxx[n-1][2] = 0.0;
    Dxx[n-1][3] = 0.0;
    return Dxx;
}
void testInitOperator(REAL* x, REAL* Dxx, const unsigned n, REAL* MyResult){
     const int block_size = 32;
    unsigned int num_blocks =  ( (n % block_size) == 0) ?
                                  n / block_size     :
                                  n / block_size + 1 ;
     unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    srand(time(NULL));
    // call CPU code
    gettimeofday(&t_start, NULL);
    MyResult = initOperator(x,Dxx);
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("Init Operator on CPU runs in: %lu microsecs", elapsed);

    // allocate memory on GPU
    float* d_x;
    cudaMalloc((void**)&d_x, n*sizeof(float));

    float* d_Dxx;
    cudaMalloc((void**)&d_Dxx, n*sizeof(float));

    // Copy data on CPU to GPU
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Dxx, Dxx, n*sizeof(float), cudaMemcpyHostToDevice);
    // perform operation
    parInitOperator(Real* x, Real* Dxx, const unsigned n)
    // copy data back to CPU

}*/
int main(int argc, char** argv) {
    const unsigned int mem_size = N * sizeof(REAL);

    // allocate arrays on CPU:    
    REAL* a = (REAL*) malloc(mem_size);
    REAL* b = (REAL*) malloc(mem_size);
    REAL* c = (REAL*) malloc(mem_size);
    REAL* r = (REAL*) malloc(mem_size);
    REAL* gpu_u  = (REAL*) malloc(mem_size);
    REAL* cpu_u  = (REAL*) malloc(mem_size);
    REAL* gpu_uu = (REAL*) malloc(mem_size);
    REAL* cpu_uu = (REAL*) malloc(mem_size);

    // init a, b, c, y
    init(BLOCK_SIZE, N, a, b, c, r);
    
    // allocate gpu arrays
    REAL *d_a, *d_b, *d_c, *d_r, *d_uu, *d_u;
    cudaMalloc((void**)&d_a,  mem_size);
    cudaMalloc((void**)&d_b,  mem_size);
    cudaMalloc((void**)&d_c,  mem_size);
    cudaMalloc((void**)&d_r,  mem_size);
    cudaMalloc((void**)&d_uu, mem_size);
    cudaMalloc((void**)&d_u,  mem_size);

    // Host-To-Device Copy
    cudaMemcpy(d_a, a, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, mem_size, cudaMemcpyHostToDevice);

    // execute on CPU
    goldenSeqTridagPar(a,b,c,r, N,SGM_SIZE, cpu_u,cpu_uu);

    // execute on GPU
    tridagCUDAWrapper(BLOCK_SIZE, d_a,d_b,d_c,d_r, N,SGM_SIZE, d_u,d_uu);
    
    cudaError_t cudaReturnCode = cudaPeekAtLastError();
    if(cudaReturnCode != cudaSuccess ) 
    {
        printf("\nCUDA ERROR: \"%i: %s\".\n", cudaReturnCode, cudaGetErrorString(cudaReturnCode));
    }

    // transfer back to CPU
    cudaMemcpy(gpu_u,   d_u,   mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_uu,  d_uu,  mem_size, cudaMemcpyDeviceToHost);
    
    // free gpu memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_r);
    cudaFree(d_u); cudaFree(d_uu);

    // validate
    validate(N, cpu_u, gpu_u);

    printf("It Amazingly Validates!!!\n\n");

    // deallocate cpu arrays
    free(a); free(b); free(c); free(r);
    free(gpu_uu); free(cpu_uu);
    free(gpu_u);  free(cpu_u);

    return 0;
}


