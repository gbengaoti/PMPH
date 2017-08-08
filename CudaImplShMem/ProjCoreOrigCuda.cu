#include <algorithm>
#include "Constants.h"
#include <stdio.h>
#include "ProjHelperFunCuda.h"
#include "ProjHelperFun.h"
#include "CudaUtilProj.cu.h"
#include "CudaKernels.cu.h"

// divide and round up
unsigned int divup(unsigned int a, unsigned int b) {
    return ((a + (b - 1)) / b);
}

// return a value that is integer divisible by aligner and not larger
// than the input
unsigned int alignNum(unsigned int numToAlign, unsigned int aligner) {
    return aligner  * (numToAlign - numToAlign % aligner) / aligner;
}


void
rollback( const unsigned            outer, 
          const unsigned            g,   
          const PrivGlobsInvCuda&   globsinvCuda,
          ExpGlobsCuda&             globsCuda,
          TempGlobsCuda&            temp_globs,
          unsigned                  numX,
          unsigned                  numY ) {

    /********************************
            CUDA Declarations
    *********************************/
    REAL *d_u = temp_globs.d_u;
    REAL *d_u_t = temp_globs.d_u_t;
    REAL *d_v = temp_globs.d_v;
    REAL *d_a = temp_globs.d_a;
    REAL *d_b = temp_globs.d_b;
    REAL *d_c = temp_globs.d_c;
    REAL *d_y = temp_globs.d_y;
    REAL *d_yy = temp_globs.d_yy;

    /********************************
            CUDA Kernel calls
    *********************************/
    dim3 threads_per_3d_block(32,32,1); //32*32 = 1024 threads total
    dim3 num_3d_blocks(divup(numX, threads_per_3d_block.x), divup(numY, threads_per_3d_block.y), divup(outer, threads_per_3d_block.z));
    dim3 num_3d_blocks_interchange(divup(numY, threads_per_3d_block.y), divup(numX, threads_per_3d_block.x), divup(outer, threads_per_3d_block.z));
    
    ExplicitXNaiveCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarX, globsCuda.d_myResult, globsinvCuda.d_myDxx, d_u_t, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    cudaThreadSynchronize();

    transpose<REAL, 32>( d_u_t, d_u, numX, numY, outer);
    ExplicitYNaiveCuda<<<num_3d_blocks_interchange,threads_per_3d_block>>>(globsCuda.d_myVarY, globsCuda.d_myResult, globsinvCuda.d_myDyy, d_u, d_v, numX, numY, outer);
    cudaThreadSynchronize();

    ImplicitXSetupNaiveCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarX, globsinvCuda.d_myDxx, d_a, d_b, d_c, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    cudaThreadSynchronize();

    // set up shared memory tridag - this is an attempt to optimize the 
    // number of threads and shared memory usage
    unsigned int maxSharedMemory = 48 * 1024;
    unsigned int maxNumRowsX = maxSharedMemory / (sizeof(REAL) * numX);
    maxNumRowsX = min<unsigned int>(128u, maxNumRowsX);
    maxNumRowsX = alignNum(maxNumRowsX, 16u);
    if (maxNumRowsX == 0)
    {
        maxNumRowsX = 1;
    }

    dim3 tridag_y_block_size(maxNumRowsX, 1, 1);
    dim3 tridag_y_blocks(divup(numY, tridag_y_block_size.x), divup(outer, tridag_y_block_size.y), 1); 

    tridagShMemCuda<<<tridag_y_blocks, tridag_y_block_size, (maxNumRowsX+1) * numX * sizeof(REAL)>>>(d_a, d_b, d_c, d_u, numX, numY, outer, d_u, d_yy);
    cudaThreadSynchronize();

    transpose<REAL, 32>( d_u, d_u_t, numY, numX, outer);
    ImplicitYSetupNaiveCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarY, globsinvCuda.d_myDyy, d_a, d_b, d_c, d_u_t, d_v, d_y, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    cudaThreadSynchronize();

    // set up shared memory tridag - this is an attempt to optimize the 
    // number of threads and shared memory usage
    unsigned int maxNumRowsY = maxSharedMemory / (sizeof(REAL) * numY);
    maxNumRowsY = min<unsigned int>(128u, maxNumRowsY);
    maxNumRowsY = alignNum(maxNumRowsY, 16u);
    if (maxNumRowsY == 0)
    {
        maxNumRowsY = 1;
    }
    dim3 tridag_x_block_size(maxNumRowsY, 1, 1);
    dim3 tridag_x_blocks(divup(numX, tridag_x_block_size.x), divup(outer, tridag_x_block_size.y), 1); 
    tridagShMemCuda<<<tridag_x_blocks, tridag_x_block_size, (maxNumRowsY+1) * numY * sizeof(REAL)>>>(d_a, d_b, d_c, d_y, numY, numX, outer, globsCuda.d_myResult, d_yy);
    cudaThreadSynchronize();
}

void   run_OrigCPU(  
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t, 
                const REAL&           alpha, 
                const REAL&           nu, 
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
) {
    printf("Dimensions: numX: %i, numY: %i, numT: %i, outer: %i\n", numX, numY, numT, outer);

    PrivGlobsInvCuda globs_inv_cuda(numX, numY, numT);
    unsigned int threads_per_block = 32; //32*32 = 1024 threads total
    unsigned int num_blocksX = divup(numX, threads_per_block);
    unsigned int num_blocksY = divup(numY, threads_per_block);
    unsigned int num_blocksT = divup(numT, threads_per_block);

    InitGlobsInvX<<<num_blocksX, threads_per_block>>>(globs_inv_cuda.d_myX, globs_inv_cuda.d_myXindex, numX, s0, alpha, nu, t);
    InitGlobsInvY<<<num_blocksY, threads_per_block>>>(globs_inv_cuda.d_myY, globs_inv_cuda.d_myYindex, numY, s0, alpha, nu, t);
    InitGlobsInvT<<<num_blocksT, threads_per_block>>>(globs_inv_cuda.d_myTimeline, numT, t);
    cudaThreadSynchronize();
    InitOperatorCuda<<<num_blocksX, threads_per_block>>>(globs_inv_cuda.d_myX, globs_inv_cuda.d_myDxx, numX);
    InitOperatorCuda<<<num_blocksY, threads_per_block>>>(globs_inv_cuda.d_myY, globs_inv_cuda.d_myDyy, numY);
    cudaThreadSynchronize();

    ExpGlobsCuda globs_cuda(numX, numY, numT);

    TempGlobsCuda temp_globs(numX, numY, numT);

    dim3 threads_per_3d_block(32,32,1); //32*32 = 1024 threads total
    dim3 num_3d_blocks(divup(numX, threads_per_3d_block.x), divup(numY, threads_per_3d_block.y), divup(outer, threads_per_3d_block.z));
    SetPayoffCuda<<<num_3d_blocks,threads_per_3d_block>>>(globs_inv_cuda.d_myX, globs_cuda.d_myResult, 0.001, numX, numY, outer);
    cudaThreadSynchronize();   

    // why this interchange is safe to do
    // why this loop cannot be parallelized
    for(int g = numT-2; g>=0; --g) {

        UpdateParamsCuda<<<num_3d_blocks,threads_per_3d_block>>>(
            globs_inv_cuda.d_myX, globs_inv_cuda.d_myY, globs_inv_cuda.d_myTimeline,
            globs_cuda.d_myVarX, globs_cuda.d_myVarY, 
            alpha, beta, nu, g, numX, numY, outer);
        cudaThreadSynchronize();

        // rollback only reads the value of i and globs but does not modify
        rollback(outer, g, globs_inv_cuda, globs_cuda, temp_globs, numX, numY);
    }
    
    REAL* d_globResults;
    cudaMalloc((void**)&d_globResults, sizeof(REAL) * outer);
    GetResults<<<divup(outer, threads_per_block), threads_per_block>>>(d_globResults, globs_cuda.d_myResult
        , globs_inv_cuda.d_myXindex, globs_inv_cuda.d_myYindex
        , numX, numY, outer);
    cudaThreadSynchronize();

    cudaMemcpy(res, d_globResults, sizeof(REAL) * outer, cudaMemcpyDeviceToHost);

    cudaFree(d_globResults);

    // did we encounter any errors?
    cudaError_t cudaReturnCode = cudaPeekAtLastError();
    if(cudaReturnCode != cudaSuccess ) 
    {
        printf("\nCUDA ERROR: \"%i: %s\".\n", cudaReturnCode, cudaGetErrorString(cudaReturnCode));
    }
}

