#ifndef PROJ_HELPER_FUNS_CUDA
#define PROJ_HELPER_FUNS_CUDA

#include <cuda_runtime.h>

//using namespace std;

// PrivGlobsInv are the invariant parts (hence, the name) 
// of the globs structure in the hand out code. 
struct PrivGlobsInvCuda {

    //  grid
    REAL*       d_myX;        // [numX]
    REAL*       d_myY;        // [numY]
    REAL*       d_myTimeline; // [numT]
    unsigned*   d_myXindex;  
    unsigned*   d_myYindex;

    //  operators
    REAL*       d_myDxx;  // [numX][4]
    REAL*       d_myDyy;  // [numY][4]

    PrivGlobsInvCuda( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobsInvCuda(   const unsigned int& numX,
                        const unsigned int& numY,
                        const unsigned int& numT ){
        cudaMalloc((void**)&d_myX,        sizeof(REAL) * numX);
        cudaMalloc((void**)&d_myDxx,      sizeof(REAL) * numX * 4u);
        cudaMalloc((void**)&d_myXindex,   sizeof(unsigned));

        cudaMalloc((void**)&d_myY,        sizeof(REAL) * numY);
        cudaMalloc((void**)&d_myDyy,      sizeof(REAL) * numY * 4u);
        cudaMalloc((void**)&d_myYindex,   sizeof(unsigned));

        cudaMalloc((void**)&d_myTimeline, sizeof(REAL) * numT);
    }

    ~PrivGlobsInvCuda( ){
        cudaFree(d_myX);
        cudaFree(d_myDxx);
        cudaFree(d_myXindex);

        cudaFree(d_myY);
        cudaFree(d_myDyy);        
        cudaFree(d_myYindex);
        
        cudaFree(d_myTimeline);
    }
};

struct ExpGlobsCuda {

    //  variable
    REAL* d_myResult; // [OUTER][numX][numY]

    //  coeffs
    REAL*  d_myVarX;  // [OUTER][numX][numY]
    REAL*  d_myVarY;  // [OUTER][numX][numY]

    ExpGlobsCuda( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    ExpGlobsCuda(   const unsigned int& numX,
                    const unsigned int& numY,
                    const unsigned int& outer ) {
        unsigned int memUsageBytes = numX * numY * outer * sizeof(REAL);

        cudaMalloc((void**)&d_myResult, memUsageBytes);
        cudaMalloc((void**)&d_myVarX,   memUsageBytes);
        cudaMalloc((void**)&d_myVarY,   memUsageBytes);
    }

    ~ExpGlobsCuda() {
        cudaFree(d_myResult);
        cudaFree(d_myVarX);
        cudaFree(d_myVarY);
    }
};


void compare(REAL* expected, REAL* d_actual, unsigned int bytes, const char* message)
{
    bool errorEncountered = false;
    cudaError_t cudaReturnCode = cudaPeekAtLastError();
    if(cudaReturnCode != cudaSuccess ) 
    {
        printf("\nCUDA ERROR: \"%i: %s\" while comparing %s.\n", cudaReturnCode, cudaGetErrorString(cudaReturnCode), message);
        errorEncountered = true;
    }

    REAL* actual = (REAL*)malloc(bytes);
    if (actual == NULL)
    {
        printf("\nCPU ERROR: Unable to allocate for %s. \n", message);
        errorEncountered = true;
    }
    cudaMemcpy(actual, d_actual, bytes, cudaMemcpyDeviceToHost);

    cudaReturnCode = cudaPeekAtLastError();
    if(cudaReturnCode != cudaSuccess ) 
    {
        printf("\nCUDA ERROR after cudaMemcpy: \"%i: %s\" while comparing %s.\n", cudaReturnCode, cudaGetErrorString(cudaReturnCode), message);
        errorEncountered = true;
    }

    for (int i = 0; i < bytes / sizeof(REAL); ++i)
    {
        if (std::abs(expected[i] - actual[i]) > 1e-4)
        {
            printf("\nERROR (%i): %s: Expected: %f, but got %f.\n", i, message, expected[i], actual[i]);
            errorEncountered = true;
            break;
        }
    }

    free(actual);

    if(errorEncountered) {
        printf("\tBailing out on run.\n");
        exit(0);
    }
}

#endif // PROJ_HELPER_FUNS_CUDA
