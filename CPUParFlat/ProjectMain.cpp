#include "OpenmpUtil.h"
#include "ParseInput.h"
 #include <sys/time.h>
#include "ProjHelperFun.h"

int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}
int main()
{
    unsigned int OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T; 
	const REAL s0 = 0.03, strike = 0.03, t = 5.0, alpha = 0.2, nu = 0.6, beta = 0.5;

    readDataSet( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T ); 

    const int Ps = get_CPU_num_threads();
    REAL* res = (REAL*)malloc(OUTER_LOOP_COUNT*sizeof(REAL));

    {   // Original Program (Sequential CPU Execution)
        cout<<"\n// Running Original, Sequential Project Program"<<endl;

        unsigned long int elapsed = 0;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        run_OrigCPU( OUTER_LOOP_COUNT, NUM_X, NUM_Y, NUM_T, s0, t, alpha, nu, beta, res );

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

        // validation and writeback of the result
        bool is_valid = validate   ( res, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, res, OUTER_LOOP_COUNT, 
                             NUM_X, NUM_Y, NUM_T, false, Ps, elapsed );        
    }

    return 0;
}

