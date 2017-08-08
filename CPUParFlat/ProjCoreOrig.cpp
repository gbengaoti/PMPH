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

    REAL* test = (REAL*)malloc(sizeof(REAL)*12);
    for(int k = 0; k < 12; k++) {
        test[k] = k+1;
        printf("%f ", test[k]);
    }

    printf("\n");

    REAL* test_t = (REAL*)malloc(sizeof(REAL)*12);

    transpose3d(test, test_t, 2, 2, 3);

    for(int j = 0; j < 12; j++) {
        printf("%f ", test_t[j]);
    }  

    printf("\n DONE \n\n\n");

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

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        #pragma omp parallel for default(shared) schedule(static)
        for(j=0;j<numY;j++) {
            tridag(    a + out*numX*numY + j*numX,
                       b + out*numX*numY + j*numX,
                       c + out*numX*numY + j*numX,
                       u + out*numX*numY + j*numX, 
                       numX, 
                       u + out*numX*numY + j*numX, 
                       yy + out*numX*numY + j*numX);
        }
    }

    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        //  implicit y
        #pragma omp parallel for default(shared) schedule(static)
        for(i=0;i<numX;i++) { 
            //#pragma omp parallel for default(shared) schedule(static)
            for(j=0;j<numY;j++) {  // here a, b, c should have size [numY]

                unsigned int loc_ij = out*numX*numY + i*numY + j;
                unsigned int loc_ji = out*numX*numY + j*numX + i;

                a[loc_ij] =       - 0.5*(0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)]);
                b[loc_ij] = dtInv - 0.5*(0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)+1]);
                c[loc_ij] =       - 0.5*(0.5*globs.myVarY[loc_ij]*globsinv.myDyy[(j*4)+2]);
                y[loc_ij] = dtInv*u[loc_ji] - 0.5*v[out*numX*numY + i*numY + j];
            } 
        }
    }

    
    #pragma omp parallel for default(shared) schedule(static)
    for( unsigned out = 0; out < outer; ++ out ) {
        #pragma omp parallel for default(shared) schedule(static)
        for(i=0;i<numX;i++) { 
            tridag(    a + out*numX*numY + i*numY,
                       b + out*numX*numY + i*numY,
                       c + out*numX*numY + i*numY,
                       y + out*numX*numY + i*numY,
                       numY,
                       &globs.myResult[out*numY*numX + i*numY],
                       yy + out*numX*numY + i*numY);
        }
    
    }
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
