#ifndef CUDA_IMPL_KERNELS
#define CUDA_IMPL_KERNELS

#include <cuda_runtime.h>


__global__ void dummyKernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

//unsigned int threads_per_block = 256; //32*32 = 1024 threads total
//unsigned int num_blocksT = divup(numT, threads_per_block);
//InitGlobsInvT<<<num_blocksT, threads_per_block>>>(globs_inv_cuda.d_myTimeline, numT, t);
__global__ void InitGlobsInvT(
    REAL* d_myTimeline, // output
    unsigned int numT,  // array dimensions
    const REAL t)   // inputs
{
    // init grid
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // global id
    if(i < numT)
    {
        d_myTimeline[i] = t*i/(numT-1);
    }
}

//unsigned int threads_per_block = 256; //32*32 = 1024 threads total
//unsigned int num_blocksX = divup(numX, threads_per_block);
//InitGlobsInvX<<<num_blocksX, threads_per_block>>>(globs_inv_cuda.d_myX, globs_inv_cuda.d_myXindex, numX, s0, alpha, nu, t);
__global__ void InitGlobsInvX(
    REAL* d_myX,       // output
    //REAL* d_myDxx,     // output
    unsigned int* d_myXindex,  // output
    unsigned int numX, //array dimensions
    REAL s0, const REAL alpha, const REAL nu, const REAL t)   // inputs
{
    // init grid
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // global id
    if (i < numX)
    {
        const REAL stdX = 20.0*alpha*s0*sqrt(t);
        const REAL dx = stdX/numX;
        const unsigned int myXindex = static_cast<unsigned>(s0/dx) % numX;

        d_myX[i] = i*dx - myXindex*dx + s0;

        if (i == 0)
        {
            *d_myXindex = myXindex;
        }
        
    }
}

//unsigned int threads_per_block = 256; //32*32 = 1024 threads total
//unsigned int num_blocksY = divup(numY, threads_per_block);
//InitGlobsInvY<<<num_blocksY, threads_per_block>>>(globs_inv_cuda.d_myY, globs_inv_cuda.d_myYindex, numY, s0, alpha, nu, t);
__global__ void InitGlobsInvY(
    REAL* d_myY,       // output
    //REAL* d_myDyy,     // output
    unsigned int* d_myYindex,  // output
    unsigned int numY, // array dimensions
    REAL s0, const REAL alpha, const REAL nu,const REAL t)   // inputs
{
    // init grid
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // global id
    if (i < numY)
    {
        const REAL stdY = 10.0*nu*sqrt(t);
        const REAL dy = stdY/numY;
        const REAL logAlpha = log(alpha);
        const unsigned int myYindex = static_cast<unsigned>(numY/2.0);

        d_myY[i] = i*dy - myYindex*dy + logAlpha;

        if (i == 0)
        {
            *d_myYindex = myYindex;
        }
    }
}

//InitOperatorCuda<<<num_blocksX, threads_per_block>>>(globs_inv_cuda.d_myX, globs_inv_cuda.d_myDxx, numX);
//InitOperatorCuda<<<num_blocksY, threads_per_block>>>(globs_inv_cuda.d_myY, globs_inv_cuda.d_myDyy, numY);
__global__ void InitOperatorCuda(
    REAL* d_x,       // input
    REAL* d_Dxx,     // output
    unsigned int n)  // array dimensions
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // global id

    if (i == 0)
    {
        d_Dxx[i*4 + 0] =  0.0;
        d_Dxx[i*4 + 1] =  0.0;
        d_Dxx[i*4 + 2] =  0.0;
        d_Dxx[i*4 + 3] =  0.0;
    }
    else if(i == n - 1)
    {
        d_Dxx[i*4 + 0] = 0.0;
        d_Dxx[i*4 + 1] = 0.0;
        d_Dxx[i*4 + 2] = 0.0;
        d_Dxx[i*4 + 3] = 0.0;
    }
    else if (i < n)
    {
        REAL x_im = d_x[i-1];
        REAL x_i  = d_x[i];
        REAL x_ip = d_x[i+1];

        REAL dxl      = x_i   - x_im;
        REAL dxu      = x_ip  - x_i;

        d_Dxx[i*4 + 0] =  2.0/dxl/(dxl+dxu);
        d_Dxx[i*4 + 1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        d_Dxx[i*4 + 2] =  2.0/dxu/(dxl+dxu);
        d_Dxx[i*4 + 3] =  0.0; 
    }
}

//dim3 threads_per_3d_block(32,32,1); //32*32 = 1024 threads total
//dim3 num_3d_blocks(divup(numX, threads_per_3d_block.x), divup(numY, threads_per_3d_block.y), divup(outer, threads_per_3d_block.z));
//SetPayoffCuda<<<num_3d_blocks,threads_per_3d_block>>>(globs_inv_cuda.d_myX, globs_cuda.d_myResult, 0.001, numX, numY, outer);
__global__ void SetPayoffCuda(
    REAL* d_x,       // input
    REAL* d_Result,  // output
    REAL strikeCoefficient,     // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    if (out < outer && j < numY && i < numX)
    {
        unsigned int loc = out*numX*numY + i*numY + j;
        // strike variable removed
        REAL x = d_x[i];
        d_Result[loc] = max(x-(strikeCoefficient * out), (REAL)0.0);
    }
}


//dim3 threads_per_3d_block(32,32,1); //32*32 = 1024 threads total
//dim3 num_3d_blocks(divup(numX, threads_per_3d_block.x), divup(numY, threads_per_3d_block.y), divup(outer, threads_per_3d_block.z));
//UpdateParamsCuda<<<num_3d_blocks,threads_per_3d_block>>>(
//            globs_inv_cuda.d_myX, globs_inv_cuda.d_myY, globs_inv_cuda.d_myTimeline,
//            globs_cuda.d_myVarX, globs_cuda.d_myVarY, 
//            alpha, beta, nu, g, numX, numY, outer);
__global__ void UpdateParamsCuda(
    REAL* d_x,          // input
    REAL* d_y,          // input
    REAL* d_timeline,   // input
    REAL* d_varX,       // output
    REAL* d_varY,       // output
    REAL alpha, REAL beta, REAL nu, unsigned int g, // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    if (out < outer && j < numY && i < numX)
    {
        unsigned int loc = out*numX*numY + i*numY + j; // use j instead of j1
        REAL x = d_x[i];
        REAL y = d_y[j];
        REAL t = d_timeline[g];

        REAL varX = exp( 2.0 * (beta * log(x)+ y - 0.5 * nu * nu * t));

        REAL varY = exp( 2.0 * (alpha * log(x)+ y - 0.5 * nu * nu * t));
        
        d_varX[loc] = varX;
        d_varY[loc] = varY;
    }
}

__global__ void ExplicitXNaiveCuda(
    REAL* d_varX,       
    REAL* d_results,
    REAL* d_dxx,
    REAL* d_u, // output
    REAL* d_timeline, 
    const int g,        // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);

    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ij = out*numX*numY + i*numY + j;

        REAL u_ji = dtInv*d_results[loc_ij];
        REAL varX_ij = d_varX[loc_ij];
        
        if(i > 0) { 
            u_ji += 0.5*( 0.5*varX_ij*d_dxx[i*4] ) 
                        * d_results[out*numX*numY + (i-1)*numY + j];

        }
        
        
        u_ji  +=  0.5*( 0.5*varX_ij*d_dxx[(i*4)+1] )
                        * d_results[loc_ij];

        if(i < numX-1) {
            u_ji += 0.5*( 0.5*varX_ij*d_dxx[(i*4)+2] )
                        * d_results[out*numX*numY + (i+1)*numY + j];
        } 

        d_u[loc_ij] =  u_ji;
    }
}

__global__ void ExplicitYNaiveCuda(
    REAL* d_varY,       // output
    REAL* d_results,
    REAL* d_dyy,
    REAL* d_u,
    REAL* d_v,
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ji = out*numX*numY + j*numX + i;
        unsigned int loc_ij = out*numX*numY + i*numY + j;

        REAL v_ij = 0.0;
        REAL varY_ij = d_varY[loc_ij];
        
        if(j > 0) {
            v_ij +=  ( 0.5 * varY_ij * d_dyy[j*4] )
                     *  d_results[loc_ij-1];
        }
        
        v_ij  +=   ( 0.5 * varY_ij * d_dyy[(j*4)+1] )
                     *  d_results[loc_ij];

        if(j < numY-1) {
            v_ij +=  ( 0.5 * varY_ij * d_dyy[(j*4)+2] )
                     *  d_results[loc_ij+1];
        }

        d_v[loc_ij] = v_ij;

        REAL u_ji = d_u[loc_ji];
        d_u[loc_ji] = u_ji + v_ij;
    }
}


__global__ void ImplicitXSetupNaiveCuda(
    REAL* d_varX,       // output
    REAL* d_dxx,
    REAL* d_a,
    REAL* d_b,
    REAL* d_c,
    REAL* d_timeline, 
    const int g,          // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);
    
    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ji = out*numY*numX + j*numX + i;
        unsigned int loc_ij = out*numY*numX + i*numY + j;

        REAL varX_ij = d_varX[loc_ij];

        d_a[loc_ji] =       - 0.5*(0.5*varX_ij*d_dxx[(i*4)]);
        d_b[loc_ji] = dtInv - 0.5*(0.5*varX_ij*d_dxx[(i*4)+1]);
        d_c[loc_ji] =       - 0.5*(0.5*varX_ij*d_dxx[(i*4)+2]);
    }
}

__global__ void ImplicitXSetupCubeCuda(
    REAL* d_varX,       // output
    REAL* d_dxx,
    REAL* d_abc,
    REAL* d_timeline, 
    const int g,          // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);
    
    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ji = out*numY*numX + j*numX + i;
        unsigned int loc_ij = out*numY*numX + i*numY + j;

        REAL varX_ij = d_varX[loc_ij];

        REAL a =       - 0.5*(0.5*varX_ij*d_dxx[(i*4)]);
        REAL b = dtInv - 0.5*(0.5*varX_ij*d_dxx[(i*4)+1]);
        REAL c =       - 0.5*(0.5*varX_ij*d_dxx[(i*4)+2]);

        d_abc[3 * loc_ji + 0] = a;
        d_abc[3 * loc_ji + 1] = b;
        d_abc[3 * loc_ji + 2] = c;
    }
}


__global__ void ImplicitYSetupCubeCuda(
    REAL* d_varY,       // output
    REAL* d_dyy,
    REAL* d_abc,
    REAL* d_u,
    REAL* d_v,
    REAL* d_y,
    REAL* d_timeline, 
    const int g,          // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);

    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ij = out*numX*numY + i*numY + j;

        REAL varY_ij = d_varY[loc_ij];

        REAL a =       - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)]);
        REAL b = dtInv - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)+1]);
        REAL c =       - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)+2]);

        d_abc[3* loc_ij + 0] = a;
        d_abc[3* loc_ij + 1] = b;
        d_abc[3* loc_ij + 2] = c;

        d_y[loc_ij] = dtInv * d_u[loc_ij] - 0.5 * d_v[loc_ij];

    }
}

__global__ void ImplicitYSetupNaiveCuda(
    REAL* d_varY,       // output
    REAL* d_dyy,
    REAL* d_a,
    REAL* d_b,
    REAL* d_c,
    REAL* d_u,
    REAL* d_v,
    REAL* d_y,
    REAL* d_timeline, 
    const int g,          // parameters
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - x
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;   // global id - y
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;   // global id - outer

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);

    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ij = out*numX*numY + i*numY + j;

        REAL varY_ij = d_varY[loc_ij];

        d_a[loc_ij] =       - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)]);
        d_b[loc_ij] = dtInv - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)+1]);
        d_c[loc_ij] =       - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)+2]);

        d_y[loc_ij] = dtInv * d_u[loc_ij] - 0.5 * d_v[loc_ij];

    }
}

// COALESCE MEMORY HERE
__global__ void  tridagNaiveCuda(
          REAL*   d_a,   // size [n]
          REAL*   d_b,   // size [n]
          REAL*   d_c,   // size [n]
          REAL*   d_r,   // size [n]
          unsigned int n, //columns
          unsigned int m, //rows
          unsigned int outer, //depth
          REAL*   d_u,   // size [n]
          REAL*   d_uu   // size [n] temporary
) {
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  // global id - m
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;   // global id - outer

    if (j < m && out < outer)
    {
        unsigned int loc = out*n*m + j*n;

        REAL   beta;
    
        REAL u_prev = d_r[loc];
        REAL uu_prev = d_b[loc];
        d_u[loc]  = u_prev;
        d_uu[loc] = uu_prev;

        int i = 0;
        for(i=1; i<n; i++) {
            unsigned int loc_i = loc + i;
            beta  = d_a[loc_i] / uu_prev;
    
            uu_prev = d_b[loc_i] - beta*d_c[loc_i-1];
            u_prev  = d_r[loc_i] - beta*u_prev;

            d_uu[loc_i] = uu_prev;
            d_u[loc_i]  = u_prev;
        }
    
        // X) this is a backward recurrence
        u_prev = u_prev / uu_prev;
        d_u[loc + n-1] = u_prev;

        for(i=n-2; i>=0; i--) {
            unsigned int loc_i = loc + i;
            u_prev = (d_u[loc_i] - d_c[loc_i] * u_prev) 
                            / d_uu[loc_i];
            d_u[loc_i] = u_prev;            
        }
    }

}

__global__ void  tridagCoalCudaForward2(
          REAL*   d_a,   // size [n]
          REAL*   d_b,   // size [n]
          REAL*   d_c,   // size [n]
          REAL*   d_r,   // size [n]
          unsigned int numY, //columns
          unsigned int numX, //rows
          unsigned int outer, //depth
          unsigned int k, //depth
          REAL*   d_u,   // size [n]
          REAL*   d_uu   // size [n] temporary
) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - m
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;   // global id - outer

    if (i < numX && out < outer)
    {
            unsigned int loc = out*numY*numX + i*numX + k;
            unsigned int locX = out*numY*numX + i*numX + k - 1;
            if(k == 0) {
                d_u[loc] = d_r[loc];
                d_uu[loc] = d_b[loc];
            } 
            else {
                REAL beta  = d_a[loc] / d_uu[locX];
    
                d_uu[loc] = d_b[loc] - beta*d_c[locX];
                d_u[loc]  = d_r [loc] - beta*d_u[locX];
            
            }
            printf("(%u, %u, %u) %u %p\n", k, i, out, threadIdx.x, &d_u[loc]);
    }
 }
    
__global__ void  tridagCoalCudaBackward2(
          REAL*   d_a,   // size [n]
          REAL*   d_b,   // size [n]
          REAL*   d_c,   // size [n]
          REAL*   d_r,   // size [n]
          unsigned int numY, //columns
          unsigned int numX, //rows
          unsigned int outer, //depth
          unsigned int k, //depth
          REAL*   d_u,   // size [n]
          REAL*   d_uu   // size [n] temporary
) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - m
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;   // global id - outer
    if (i < numX && out < outer)
    {
            unsigned int loc = out*numY*numX + i*numX + k;
            unsigned int locX = out*numY*numX + i*numX + k + 1;
            if(k == (numY-1)) {
                REAL uk = d_u[loc];
                d_u[loc] = uk /  d_uu[loc];
            }
            else {
                REAL ck = d_c[loc];
                REAL uk = d_u[loc];
                d_u[loc] = (uk - ck*d_u[locX]) / d_uu[loc];
            }
    }

}

__global__ void  tridagCoalCudaForward(
          REAL*   d_a,   // size [n]
          REAL*   d_b,   // size [n]
          REAL*   d_c,   // size [n]
          REAL*   d_r,   // size [n]
          unsigned int numY, //columns
          unsigned int numX, //rows
          unsigned int outer, //depth
          unsigned int k, //depth
          REAL*   d_u,   // size [n]
          REAL*   d_uu   // size [n] temporary
) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - m
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;   // global id - outer

    if (i < numX && out < outer)
    {
            unsigned int loc = out*numY*numX + i*numY + k;
            unsigned int locX = out*numY*numX + i*numY + k - 1;
            if(k == 0) {
                d_u[loc] = d_r[loc];
                d_uu[loc] = d_b[loc];
            } 
            else {
                REAL beta  = d_a[loc] / d_uu[locX];
    
                d_uu[loc] = d_b[loc] - beta*d_c[locX];
                d_u[loc]  = d_r [loc] - beta*d_u[locX];
            
            }
            //printf("(%u, %u, %u) %u %p\n", k, i, out, threadIdx.x, &d_u[loc]);
    }
 }
    
__global__ void  tridagCoalCudaBackward(
          REAL*   d_a,   // size [n]
          REAL*   d_b,   // size [n]
          REAL*   d_c,   // size [n]
          REAL*   d_r,   // size [n]
          unsigned int numY, //columns
          unsigned int numX, //rows
          unsigned int outer, //depth
          unsigned int k, //depth
          REAL*   d_u,   // size [n]
          REAL*   d_uu   // size [n] temporary
) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  // global id - m
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;   // global id - outer
    if (i < numX && out < outer)
    {
            unsigned int loc = out*numY*numX + i*numY + k;
            unsigned int locX = out*numY*numX + i*numY + k + 1;
            if(k == (numY-1)) {
                REAL uk = d_u[loc];
                d_u[loc] = uk /  d_uu[loc];
            }
            else {
                REAL ck = d_c[loc];
                REAL uk = d_u[loc];
                d_u[loc] = (uk - ck*d_u[locX]) / d_uu[loc];
            }
    }

}

__global__ void  tridagCubeCuda(
          REAL*   d_abc,
          REAL*   d_r,   // size [n]
          unsigned int n, //columns
          unsigned int m, //rows
          unsigned int outer, //depth
          REAL*   d_u,   // size [n]
          REAL*   d_uu   // size [n] temporary
) {
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  // global id - m
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;   // global id - outer

    if (j < m && out < outer)
    {
        unsigned int loc = out*n*m + j*n;

        REAL a = 0;
        REAL b = d_abc[3 * loc + 1];
        REAL c = 0;
        REAL   beta;
    
        REAL u_prev = d_r[loc];
        REAL uu_prev = b;
        d_u[loc]  = u_prev;
        d_uu[loc] = uu_prev;

        int i = 0;
        for(i=1; i<n; i++) {
            unsigned int loc_i = loc + i;
            c = d_abc[3 * (loc_i-1) + 2];
            a = d_abc[3 * loc_i];
            b = d_abc[3 * loc_i + 1];

            beta  = a / uu_prev;
    
            uu_prev = b - beta*c;
            u_prev  = d_r[loc_i] - beta*u_prev;

            d_uu[loc_i] = uu_prev;
            d_u[loc_i]  = u_prev;
        }
    
        // X) this is a backward recurrence
        u_prev = u_prev / uu_prev;
        d_u[loc + n-1] = u_prev;

        for(i=n-2; i>=0; i--) {
            unsigned int loc_i = loc + i;
            c = d_abc[3 * loc_i + 2];
            u_prev = (d_u[loc_i] - c * u_prev) 
                            / d_uu[loc_i];
            d_u[loc_i] = u_prev;            
        }
    }

}

__global__ void GetResults(
    REAL* d_globResults, // output
    REAL* d_results,
    unsigned int* d_xindex,
    unsigned int* d_yindex,
    unsigned int numX, unsigned int numY, unsigned int outer)  // array dimensions
{
    // init grid
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; // global id
    if(i < outer)
    {
        unsigned int xindex = *d_xindex;
        unsigned int yindex = *d_yindex;
        d_globResults[i] = d_results[i*numX*numY + xindex*numY + yindex];
        //printf("%i %f\n", i, d_globResults[i]);
    }
}

#endif //CUDA_IMPL_KERNELS

