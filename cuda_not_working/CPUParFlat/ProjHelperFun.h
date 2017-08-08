#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Constants.h"

using namespace std;

struct PrivGlobsInv {

    //  grid
    REAL*       myX;        // [numX]
    REAL*       myY;        // [numY]
    REAL*       myTimeline; // [numT]
    unsigned    myXindex;  
    unsigned    myYindex;

    //  operators
    REAL*   myDxx;  // [numX][4]
    REAL*   myDyy;  // [numY][4]

    PrivGlobsInv( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobsInv(   const unsigned int& numX,
                        const unsigned int& numY,
                        const unsigned int& numT ) {

        myX        = (REAL*)malloc(sizeof(REAL)*numX);
        myY        = (REAL*)malloc(sizeof(REAL)*numY);
        myTimeline = (REAL*)malloc(sizeof(REAL)*numT);
        myDxx      = (REAL*)malloc(sizeof(REAL)*numX*4);
        myDyy      = (REAL*)malloc(sizeof(REAL)*numY*4);
        myXindex   = 0;
        myYindex   = 0;

    }
    ~PrivGlobsInv () {
        free(myX);
        free(myY);
        free(myTimeline);
        free(myDxx);
        free(myDyy);
    }   

}
// NVCC does not support the aligned attribute
#ifdef __NVCC__
;
#else
 __attribute__ ((aligned (128)));
#endif

struct ExpGlobs {

    //  variable
    REAL* myResult; // [OUTER][numX][numY]
    REAL* myResult_t;

    //  coeffs
    REAL*  myVarX; // [OUTER][numX][numY]
    REAL*  myVarY; // [OUTER][numX][numY]

    ExpGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    ExpGlobs(  const unsigned int& numX,
               const unsigned int& numY,
               const unsigned int& outer ) {

        unsigned int struct_size = outer*numX*numY*sizeof(REAL);

        myVarX   = (REAL*)malloc(struct_size);
        myVarY   = (REAL*)malloc(struct_size);
        myResult = (REAL*)malloc(struct_size);
        myResult_t = (REAL*)malloc(struct_size);

    }

    ~ExpGlobs() {
        free(myVarX);
        free(myVarY);
        free(myResult);
        free(myResult_t);
    }
}
// NVCC does not support the aligned attribute
#ifdef __NVCC__
;
#else
 __attribute__ ((aligned (128)));
#endif



void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobsInv& globs    
            );

void initOperator(  const REAL* x, 
                              REAL* Dxx,
                              unsigned int n
                 );

void run_OrigCPU(  
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
            );

#endif // PROJ_HELPER_FUNS
