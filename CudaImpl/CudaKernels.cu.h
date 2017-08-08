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

__global__ void InitGlobsInvT(
    REAL* d_myTimeline,
    unsigned int numT,
    const REAL t)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numT)
    {
        d_myTimeline[i] = t*i/(numT-1);
    }
}

__global__ void InitGlobsInvX(
    REAL* d_myX,
    unsigned int* d_myXindex,
    unsigned int numX,
    REAL s0, const REAL alpha, const REAL nu, const REAL t)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < numX){
        const REAL stdX = 20.0*alpha*s0*sqrt(t);
        const REAL dx = stdX/numX;
        const unsigned int myXindex = static_cast<unsigned>(s0/dx) % numX;

        d_myX[i] = i*dx - myXindex*dx + s0;

        if (i == 0){
            *d_myXindex = myXindex;
        }
        
    }
}

__global__ void InitGlobsInvY(
    REAL* d_myY,
    unsigned int* d_myYindex,
    unsigned int numY,
    REAL s0, const REAL alpha, const REAL nu,const REAL t)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < numY) {
        const REAL stdY = 10.0*nu*sqrt(t);
        const REAL dy = stdY/numY;
        const REAL logAlpha = log(alpha);
        const unsigned int myYindex = static_cast<unsigned>(numY/2.0);

        d_myY[i] = i*dy - myYindex*dy + logAlpha;

        if (i == 0) {
            *d_myYindex = myYindex;
        }
    }
}

__global__ void InitOperatorCuda(
    REAL* d_x,
    REAL* d_Dxx,
    unsigned int n) {

    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

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

__global__ void SetPayoffCuda(
    REAL* d_x,
    REAL* d_Result,
    REAL strikeCoefficient,
    unsigned int numX, unsigned int numY, unsigned int outer)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;

    if (out < outer && j < numY && i < numX)
    {
        unsigned int loc = out*numX*numY + i*numY + j;
        REAL x = d_x[i];
        d_Result[loc] = max(x-(strikeCoefficient * out), (REAL)0.0);
    }
}

__global__ void UpdateParamsCuda(
    REAL* d_x,
    REAL* d_y,
    REAL* d_timeline,
    REAL* d_varX,
    REAL* d_varY,
    REAL alpha, REAL beta, REAL nu, unsigned int g,
    unsigned int numX, unsigned int numY, unsigned int outer)
{
    const unsigned int i1 = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y; 
    const unsigned int i = blockIdx.z*blockDim.z + threadIdx.z; 

    if (i < outer && j < numY && i1 < numX)
    {
        unsigned int loc = i*numX*numY + i1*numY + j; 
        REAL x = d_x[i1];
        REAL y = d_y[j];
        REAL t = d_timeline[g];

        REAL varX = exp( 2.0 * (beta * log(x)+ y - 0.5 * nu * nu * t));

        REAL varY = exp( 2.0 * (alpha * log(x)+ y - 0.5 * nu * nu * t));
        
        d_varX[loc] = varX;
        d_varY[loc] = varY;
    }
}

__global__ void ExplicitXOptCuda(
    REAL* d_varX,       
    REAL* d_results,
    REAL* d_dxx,
    REAL* d_u,
    REAL* d_timeline, 
    const int g,
    unsigned int numX, unsigned int numY, unsigned int outer)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;  
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;

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

__global__ void ExplicitYOptCuda(
    REAL* d_varY,
    REAL* d_results,
    REAL* d_dyy,
    REAL* d_u,
    REAL* d_v,
    unsigned int numX, unsigned int numY, unsigned int outer)
{
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  
    const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;  
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;

    if (i < numX && j < numY && out < outer)
    {
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

        REAL u_ji = d_u[loc_ij];
        d_u[loc_ij] = u_ji + v_ij;
    }
}


__global__ void ImplicitXOptCuda(
    REAL* d_varX,
    REAL* d_dxx,
    REAL* d_a,
    REAL* d_b,
    REAL* d_c,
    REAL* d_timeline, 
    const int g, 
    unsigned int numX, unsigned int numY, unsigned int outer)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  
    const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;  
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);
    
    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ij = out*numY*numX + i*numY + j;

        REAL varX_ij = d_varX[loc_ij];

        d_a[loc_ij] =       - 0.5*(0.5*varX_ij*d_dxx[(i*4)]);
        d_b[loc_ij] = dtInv - 0.5*(0.5*varX_ij*d_dxx[(i*4)+1]);
        d_c[loc_ij] =       - 0.5*(0.5*varX_ij*d_dxx[(i*4)+2]);
    }
}

__global__ void ImplicitYOptCuda(
    REAL* d_varY,
    REAL* d_dyy,
    REAL* d_a,
    REAL* d_b,
    REAL* d_c,
    REAL* d_u,
    REAL* d_v,
    REAL* d_y,
    REAL* d_timeline, 
    const int g,
    unsigned int numX, unsigned int numY, unsigned int outer)
{
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;  
    const unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;  
    const unsigned int out = blockIdx.z*blockDim.z + threadIdx.z;

    REAL dtInv  = 1.0/(d_timeline[g+1]-d_timeline[g]);

    if (i < numX && j < numY && out < outer)
    {
        unsigned int loc_ij = out*numX*numY + i*numY + j;
        unsigned int loc_ji = out*numX*numY + j*numX + i;

        REAL varY_ij = d_varY[loc_ij];

        d_a[loc_ji] =       - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)]);
        d_b[loc_ji] = dtInv - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)+1]);
        d_c[loc_ji] =       - 0.5 * (0.5 * varY_ij * d_dyy[(j*4)+2]);

        d_y[loc_ji] = dtInv * d_u[loc_ij] - 0.5 * d_v[loc_ij];

    }
}

__global__ void  tridagOptCudaY(
          REAL*   d_a,
          REAL*   d_b,
          REAL*   d_c,
          unsigned int numX,
          unsigned int numY,
          unsigned int outer, 
          REAL*   d_u,
          REAL*   d_uu
) {
    const unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;

    if (j < numY && out < outer)
    {

        for(int k=0; k<numX; k++) {

            unsigned int loc = out*numX*numY + k*numY + j;
            unsigned int locX = out*numX*numY + (k - 1)*numY + j;

            if(k == 0) {
                d_uu[loc] = d_b[loc];
            }
            else {
                REAL beta  = d_a[loc] / d_uu[locX];
                d_uu[loc] = d_b[loc] - beta*d_c[locX];
                d_u[loc]  = d_u[loc] - beta*d_u[locX];
            }
                
        }

        for(int k=numX-1; k>=0; k--) {

            unsigned int loc = out*numX*numY + k*numY + j;

            if(k == numX-1) {
                d_u[loc] = d_u[loc] / d_uu[loc];
            }
            else {
                d_u[loc] = (d_u[loc] - d_c[loc]*d_u[out*numX*numY + (k+1)*numY + j]) / d_uu[loc];
            }   
        }
    }

}

__global__ void  tridagOptCudaX(
          REAL*   d_a,
          REAL*   d_b,
          REAL*   d_c,
          REAL*   d_r,
          unsigned int numY, 
          unsigned int numX, 
          unsigned int outer,
          REAL*   d_u,
          REAL*   d_uu
) {
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;  
    const unsigned int out = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < numX && out < outer) {
        for(int k=0; k<numY; k++) { 
            unsigned int loc = out*numY*numX + k*numX + i;
            unsigned int locX = out*numY*numX + (k-1)*numX + i;
            if(k == 0) {
                d_u[loc] = d_r[loc];
                d_uu[loc] = d_b[loc];
            } 
            else {
                REAL beta  = d_a[loc] / d_uu[locX];
                d_uu[loc] =d_b[loc] - beta*d_c[locX];
                d_u[loc]  = d_r [loc] - beta*d_u[locX];
        
            }
        }

        for(int k=numY-1; k>=0; k--) {
            if(k == (numY-1)) {
                REAL uk = d_u[out*numY*numX + k*numX + i];
                d_u[out*numY*numX + k*numX + i] = uk /  d_uu[out*numX*numY + k*numX + i];
            }
            else {
                REAL ck = d_c[out*numX*numY + k*numX + i];
                REAL uk = d_u[out*numY*numX + k*numX + i];
                d_u[out*numY*numX + k*numX + i] = (uk - ck*d_u[out*numY*numX + (k+1)*numX + i]) / d_uu[out*numX*numY + k*numX + i];
            }
        }
    }

}

__global__ void GetResults(
    REAL* d_globResults,
    REAL* d_results,
    unsigned int* d_xindex,
    unsigned int* d_yindex,
    unsigned int numX, unsigned int numY, unsigned int outer
) {

    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < outer)
    {
        unsigned int xindex = *d_xindex;
        unsigned int yindex = *d_yindex;
        d_globResults[i] = d_results[i*numX*numY + xindex*numY + yindex];

    }
}

#endif 

