#include <algorithm>
#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagPar.h"

inline void tridag(
    const REAL*   a,   // size [n]
    const REAL*   b,   // size [n]
    const REAL*   c,   // size [n]
    const REAL*   r,   // size [n]
    const int     n,
          REAL*   u,   // size [n]
          REAL*   uu   // size [n] temporary
) {
    int    i, offset;
    REAL   beta;

    u[0]  = r[0];
    uu[0] = b[0];

    for(i=1; i<n; i++) {
        beta  = a[i] / uu[i-1];

        uu[i] = b[i] - beta*c[i-1];
        u[i]  = r[i] - beta*u[i-1];
    }

#if 1
    // X) this is a backward recurrence
    u[n-1] = u[n-1] / uu[n-1];
    for(i=n-2; i>=0; i--) {
        u[i] = (u[i] - c[i]*u[i+1]) / uu[i];
    }
#else
    // Hint: X) can be written smth like (once you make a non-constant)
    for(i=0; i<n; i++) a[i] =  u[n-1-i];
    a[0] = a[0] / uu[n-1];
    for(i=1; i<n; i++) a[i] = (a[i] - c[n-1-i]*a[i-1]) / uu[n-1-i];
    for(i=0; i<n; i++) u[i] = a[n-1-i];
#endif
}

void transpose3d(REAL* a, REAL* a_t, unsigned o, unsigned m, unsigned n) {

    for (int k = 0;k < o;k++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                a_t[k*m*n+j*m+i] = a[k*m*n+i*n+j];   
}


// REAL* test = (REAL*)malloc(sizeof(REAL)*12);
//     for(int k = 0; k < 12; k++) {
//         test[k] = k+1;
//         printf("%f ", test[k]);
//     }

//     printf("\n");

//     test = transpose3d(test, 2, 2, 3);

//     for(int j = 0; j < 12; j++) {
//         printf("%f ", test[j]);
//     }  

//     printf("\n DONE \n\n\n");

void
rollback( const unsigned            outer, 
          const unsigned            g,  
          const PrivGlobsInv&       globsinv, 
          ExpGlobs&                 globs,
          unsigned                  numX,
          unsigned                  numY ) {

    unsigned i, j;

    REAL dtInv  = 1.0/(globsinv.myTimeline[g+1]-globsinv.myTimeline[g]);

    unsigned int struct_size = outer*numX*numY*sizeof(REAL);

    REAL *u = (REAL*)malloc(struct_size);
    REAL *v = (REAL*)malloc(struct_size);
    REAL *a = (REAL*)malloc(struct_size);
    REAL *b = (REAL*)malloc(struct_size);
    REAL *c = (REAL*)malloc(struct_size);
    REAL *y = (REAL*)malloc(struct_size);
    REAL *yy= (REAL*)malloc(struct_size);


    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        // explicit x
        #pragma omp parallel for default(shared) schedule(static)
        for(i=0;i<numX;i++) {
            #pragma omp parallel for default(shared) schedule(static)
            for(j=0;j<numY;j++) {
                unsigned int loc_ji = out*numX*numY + j*numX + i;
                unsigned int loc_ij = out*numX*numY + i*numY + j;

                u[loc_ji] = dtInv*globs.myResult[loc_ij];
    
                if(i > 0) { 
                    u[loc_ji] += 0.5*( 0.5*globs.myVarX[loc_ij]*globsinv.myDxx[i*4] ) 
                                * globs.myResult[out*numX*numY + (i-1)*numY + j];

                }
                u[loc_ji]  +=  0.5*( 0.5*globs.myVarX[loc_ij]*globsinv.myDxx[(i*4)+1] )
                                * globs.myResult[loc_ij];

                if(i < numX-1) {
                    u[loc_ji] += 0.5*( 0.5*globs.myVarX[loc_ij]*globsinv.myDxx[(i*4)+2] )
                                * globs.myResult[out*numX*numY + (i+1)*numY + j];
                }
            }
        }

    }

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        //  explicit y
        #pragma omp parallel for default(shared) schedule(static)
        for(j=0;j<numY;j++)
        {
            #pragma omp parallel for default(shared) schedule(static)
            for(i=0;i<numX;i++) {

                unsigned int loc_ij = out*numX*numY + i*numY + j;

                v[loc_ij] = 0.0;

                if(j > 0) {
                    v[loc_ij] +=  ( 0.5*globs.myVarY[loc_ij]*globsinv.myDyy[j*4] )
                             *  globs.myResult[loc_ij-1];
                }
                v[loc_ij]  +=   ( 0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)+1] )
                             *  globs.myResult[loc_ij];

                if(j < numY-1) {
                    v[loc_ij] +=  ( 0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)+2] )
                             *  globs.myResult[loc_ij+1];
                }
                u[out*numX*numY + j*numX + i] += v[loc_ij]; 
            }
        }
    }

    

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        //  implicit x
        #pragma omp parallel for default(shared) schedule(static)
        for(j=0;j<numY;j++) {
            #pragma omp parallel for default(shared) schedule(static)
            for(i=0;i<numX;i++) {

                unsigned int loc_ji = out*numY*numX + j*numX + i;
                unsigned int loc_ij = out*numY*numX + i*numY + j;

                a[loc_ji] =       - 0.5*(0.5*globs.myVarX[loc_ij]*globsinv.myDxx[(i*4)]);
                b[loc_ji] = dtInv - 0.5*(0.5*globs.myVarX[loc_ij]*globsinv.myDxx[(i*4)+1]);
                c[loc_ji] =       - 0.5*(0.5*globs.myVarX[loc_ij]*globsinv.myDxx[(i*4)+2]);

            }
        }
    }


    REAL* y_t = (REAL*)malloc(sizeof(REAL)*outer*numX*numY);
    REAL* b_t = (REAL*)malloc(sizeof(REAL)*outer*numX*numY);
    REAL* yy_t = (REAL*)malloc(sizeof(REAL)*outer*numX*numY);
    REAL* a_t = (REAL*)malloc(sizeof(REAL)*outer*numX*numY);
    REAL* c_t = (REAL*)malloc(sizeof(REAL)*outer*numX*numY);
    REAL* u_t = (REAL*)malloc(sizeof(REAL)*outer*numX*numY);


    transpose3d(y, y_t, outer, numX, numY);
    transpose3d(b, b_t, outer, numX, numY);
    transpose3d(yy, yy_t, outer, numX, numY);
    transpose3d(a, a_t, outer, numX, numY);
    transpose3d(c, c_t, outer, numX, numY);
    transpose3d(u, u_t, outer, numX, numY);

    // TRIDAG 1 HERE

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        #pragma omp parallel for default(shared) schedule(static)
        for(j=0;j<numY;j++) {
            for(int k=0; k<numX; k++) {

                unsigned int loc = out*numX*numY + k*numY + j;
                unsigned int locX = out*numX*numY + (k-1)*numY + j;

                if(k == 0) {
                    yy_t[loc] = b_t[loc];
                }
                else {
                    REAL beta  = a_t[loc] / yy_t[locX];
                    yy_t[loc] = b_t[loc] - beta*c_t[locX];
                    u_t[loc]  = u_t[loc] - beta*u_t[locX];
                }
                
            }
        }
    }

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        #pragma omp parallel for default(shared) schedule(static)
        for(j=0;j<numY;j++) {
            
            for(int k=numX-1; k>=0; k--) {

                unsigned int loc = out*numX*numY + k*numY + j;

                if(k == numX-1) {
                    u_t[loc] = u_t[loc] / yy_t[loc];
                }
                else {
                    u_t[loc] = (u_t[loc] - c_t[loc]*u_t[loc + 1]) / yy_t[loc];
                }
                
            }
        
        }
    }

    // transpose3d(y_t, y, outer, numY, numX);
    // transpose3d(b_t, b, outer, numY, numX);
    // transpose3d(yy_t, yy, outer, numY, numX);
    // transpose3d(a_t, a, outer, numY, numX);
    // transpose3d(c_t, c, outer, numY, numX);
    // transpose3d(u_t, u, outer, numY, numX);

    // transpose3d(y, y_t, outer, numX, numY);
    // transpose3d(b, b_t, outer, numX, numY);
    // transpose3d(yy, yy_t, outer, numX, numY);
    // transpose3d(a, a_t, outer, numX, numY);
    // transpose3d(c, c_t, outer, numX, numY);
    transpose3d(globs.myResult, globs.myResult_t, outer, numX, numY);

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        //  implicit y
        #pragma omp parallel for default(shared) schedule(static)
        for(i=0;i<numX;i++) { 
            #pragma omp parallel for default(shared) schedule(static)
            for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]

                unsigned int loc_ij = out*numX*numY + i*numY + j;
                unsigned int loc_ji = out*numX*numY + j*numX + i;

                a_t[loc_ji] =       - 0.5*(0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)]);
                b_t[loc_ji] = dtInv - 0.5*(0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)+1]);
                c_t[loc_ji] =       - 0.5*(0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)+2]);
                y_t[loc_ji] = dtInv*u_t[loc_ij] - 0.5*v[loc_ij];
            } 
        }
    }

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        #pragma omp parallel for default(shared) schedule(static)
        for(i=0;i<numX;i++) {
            for(int k=0; k<numY; k++) { 
                unsigned int loc = out*numY*numX + k*numX + i;
                unsigned int locX = out*numY*numX + (k-1)*numX + i;
                if(k == 0) {
                    globs.myResult_t[loc] = y_t[loc];
                    yy_t[loc] = b_t[loc];
                } 
                else {
                    REAL beta  = a_t[loc] / yy_t[locX];

                    yy_t[loc] = b_t[loc] - beta*c_t[locX];
                    globs.myResult_t[loc]  = y_t [loc] - beta*globs.myResult_t[locX];
            
                }
            }
        }
    }
    
    
    #pragma omp parallel for default(shared) schedule(static)
    for(unsigned out = 0; out < outer; ++ out ) { 
        // X) this is a backward recurrence
        #pragma omp parallel for default(shared) schedule(static)
        for(i=0;i<numX;i++) {
            for(int k=numY-1; k>=0; k--) {
                if(k == (numY-1)) {
                    REAL uk = globs.myResult_t[out*numY*numX + k*numX + i];
                    globs.myResult_t[out*numY*numX + k*numX + i] = uk /  yy_t[out*numX*numY + k*numX + i];
                }
                else {
                    REAL ck = c_t[out*numX*numY + k*numX + i];
                    REAL uk = globs.myResult_t[out*numY*numX + k*numX + i];
                    globs.myResult_t[out*numY*numX + k*numX + i] = (uk - ck*globs.myResult_t[out*numY*numX + (k+1)*numX + i]) / yy_t[out*numX*numY + k*numX + i];
                }
            }
        }

    }

    transpose3d(globs.myResult_t, globs.myResult, outer, numY, numX);

    free(y_t);
    free(b_t);
    free(yy_t);
    free(a_t);
    free(c_t);

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

    // inits here because they are independent of the outer loop
    PrivGlobsInv globs_inv(numX, numY, numT);

    initGrid(s0,alpha,nu,t, numX, numY, numT, globs_inv);

    initOperator(globs_inv.myX,globs_inv.myDxx, numX);
    initOperator(globs_inv.myY,globs_inv.myDyy, numY);

   
    ExpGlobs  globs(numX, numY, numT);
    // why is it safe to do this distribution
    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned i = 0; i < outer; ++ i ) {
        // move strike declaration inside for privitisation
        
        // same for setPayoff
        // setPayoff modifies globs.myResult and read globs.myX,globs.myY
        #pragma omp parallel for default(shared) schedule(static)
        for(unsigned i1=0; i1<numX; ++i1) {
            #pragma omp parallel for default(shared) schedule(static)
            for(unsigned j=0; j<numY; ++j) {
                globs.myResult[i*numX*numY + i1*numY + j] = max(globs_inv.myX[i1]-(0.001*i), (REAL)0.0);
            }
        }
    }
    
    // why this interchange is safe to do
    // why this loop cannot be parallelized
    for(int g = numT-2; g>=0; --g) {
        #pragma omp parallel for default(shared) schedule(static)
        for( unsigned i = 0; i < outer; ++ i ) {
            #pragma omp parallel for default(shared) schedule(static)
            for(unsigned i1=0; i1<numX; ++i1)
                #pragma omp parallel for default(shared) schedule(static)
                for(unsigned j1=0; j1<numY; ++j1) {

                    globs.myVarX[i*numX*numY + i1*numY + j1] = exp(2.0*(  beta*log(globs_inv.myX[i1])   
                                                  + globs_inv.myY[j1]             
                                                  - 0.5*nu*nu*globs_inv.myTimeline[g] )
                                            );
                    globs.myVarY[i*numX*numY + i1*numY + j1] = exp(2.0*(  alpha*log(globs_inv.myX[i1])   
                                                  + globs_inv.myY[j1]             
                                                  - 0.5*nu*nu*globs_inv.myTimeline[g] )
                                            );
            }
        }
        
        // rollback only reads the value of i and globs but does not modify
        rollback(outer, g, globs_inv, globs, numX, numY);
    }

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned i = 0; i < outer; ++ i ) {
        res[i] = globs.myResult[i*numX*numY + globs_inv.myXindex*numY + globs_inv.myYindex];
    }
}

//#endif // PROJ_CORE_ORIG
