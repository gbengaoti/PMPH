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

void
rollback( const unsigned            outer, 
          const unsigned            g,   
          const PrivGlobsInvCuda&   globsinvCuda,
          ExpGlobsCuda&             globsCuda,
          TempGlobsCuda&            temp_globs,
          unsigned                  numX,
          unsigned                  numY ) {

    //unsigned int struct_size = outer*numX*numY*sizeof(REAL);

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
    //REAL *d_abc = temp_globs.d_abc;

    /********************************
            CUDA Kernel calls
    *********************************/
    dim3 threads_per_3d_block(32,32,1); //32*32 = 1024 threads total
    dim3 num_3d_blocks(divup(numX, threads_per_3d_block.x), divup(numY, threads_per_3d_block.y), divup(outer, threads_per_3d_block.z));
    dim3 num_3d_blocks_interchange(divup(numY, threads_per_3d_block.y), divup(numX, threads_per_3d_block.x), divup(outer, threads_per_3d_block.z));
    
    ExplicitXNaiveCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarX, globsCuda.d_myResult, globsinvCuda.d_myDxx, d_u_t, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    cudaThreadSynchronize();

    // TODO: Transpose u
    transpose<REAL, 32>( d_u_t, d_u, numX, numY, outer);

    cudaError_t cudaReturnCode = cudaPeekAtLastError();
    if(cudaReturnCode != cudaSuccess ) 
    {
        printf("\nCUDA ERROR: \"%i: %s\".\n", cudaReturnCode, cudaGetErrorString(cudaReturnCode));
    }

    // TODO: loop interchange
    ExplicitYNaiveCuda<<<num_3d_blocks_interchange,threads_per_3d_block>>>(globsCuda.d_myVarY, globsCuda.d_myResult, globsinvCuda.d_myDyy, d_u, d_v, numX, numY, outer);
    cudaThreadSynchronize();

    // TODO: Transpose myVarX
    //ImplicitXSetupCubeCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarX, globsinvCuda.d_myDxx, d_abc, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    ImplicitXSetupNaiveCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarX, globsinvCuda.d_myDxx, d_a, d_b, d_c, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    cudaThreadSynchronize();

    dim3 tridag_y_block_size(1024, 1, 1);
    dim3 tridag_y_blocks(divup(numY, tridag_y_block_size.x), divup(outer, tridag_y_block_size.y), 1); 

    //tridagCubeCuda<<<tridag_y_blocks, tridag_y_block_size>>>(d_abc, d_u, numX, numY, outer, d_u, d_yy);
    //tridagNaiveCuda<<<tridag_y_blocks, tridag_y_block_size>>>(d_a, d_b, d_c, d_u, numX, numY, outer, d_u, d_yy);
    dim3 tridag_fb_block_size2(32, 32, 1);
    dim3 tridag_fb_blocks2(divup(numY, tridag_fb_block_size2.x), divup(outer, tridag_fb_block_size2.y), 1); 
    for(int k=1; k<numX; k++) { 
        tridagCoalCudaForward2<<<tridag_fb_blocks2, tridag_fb_block_size2>>>(d_a, d_b, d_c, d_u, numY, numX, outer, k, d_u, d_yy);
        cudaThreadSynchronize();
    }
    
    for(int k=numX-1; k>=0; k--) {
        tridagCoalCudaBackward2<<<tridag_fb_blocks2, tridag_fb_block_size2>>>(d_a, d_b, d_c, d_u, numY, numX, outer, k, d_u, d_yy);
        cudaThreadSynchronize();
    }
    //tridagCoalCuda<<<tridag_y_blocks, tridag_y_block_size>>>(d_a, d_b, d_c, d_u, numX, numY, outer, d_u, d_yy);
    cudaThreadSynchronize();
    exit(0);
    // TODO: transpose u (we have already done it up there for ExplicitXNaiveCuda)
    transpose<REAL, 32>( d_u, d_u_t, numY, numX, outer);
    ImplicitYSetupNaiveCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarY, globsinvCuda.d_myDyy, d_a, d_b, d_c, d_u_t, d_v, d_y, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    //ImplicitYSetupCubeCuda<<<num_3d_blocks,threads_per_3d_block>>>(globsCuda.d_myVarY, globsinvCuda.d_myDyy, d_abc, d_u_t, d_v, d_y, globsinvCuda.d_myTimeline, g, numX, numY, outer);
    cudaThreadSynchronize();

    dim3 tridag_fb_block_size(32, 32, 1);
    dim3 tridag_fb_blocks(divup(numX, tridag_fb_block_size.x), divup(outer, tridag_fb_block_size.y), 1); 

    dim3 tridag_x_block_size(1024, 1);
    dim3 tridag_x_blocks(divup(numX, tridag_x_block_size.x), divup(outer, tridag_x_block_size.y), 1); 

    //tridagCubeCuda<<<tridag_x_blocks, tridag_x_block_size>>>(d_abc, d_y, numY, numX, outer, globsCuda.d_myResult, d_yy);
    tridagNaiveCuda<<<tridag_x_blocks, tridag_x_block_size>>>(d_a, d_b, d_c, d_y, numY, numX, outer, globsCuda.d_myResult, d_yy);
    // for(int k=1; k<numY; k++) { 
    //     tridagCoalCudaForward<<<tridag_fb_blocks, tridag_fb_block_size>>>(d_a, d_b, d_c, d_y, numY, numX, outer, k, globsCuda.d_myResult, d_yy);
    //     cudaThreadSynchronize();
    // }
    
    // for(int k=numY-1; k>=0; k--) {
    //     tridagCoalCudaBackward<<<tridag_fb_blocks, tridag_fb_block_size>>>(d_a, d_b, d_c, d_y, numY, numX, outer, k, globsCuda.d_myResult, d_yy);
    //     cudaThreadSynchronize();
    // }
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
    dim3 num_3d_blocks_test(divup(numY, threads_per_3d_block.y), divup(numX, threads_per_3d_block.x), divup(outer, threads_per_3d_block.z));
    SetPayoffCuda<<<num_3d_blocks_test,threads_per_3d_block>>>(globs_inv_cuda.d_myX, globs_cuda.d_myResult, 0.001, numX, numY, outer);
    cudaThreadSynchronize();   

    // why this interchange is safe to do
    // why this loop cannot be parallelized
    for(int g = numT-2; g>=0; --g) {

        UpdateParamsCuda<<<num_3d_blocks_test,threads_per_3d_block>>>(
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
}

