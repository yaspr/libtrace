//
#define _GNU_SOURCE

//
#include <time.h>   //time, localtime
#include <dlfcn.h>  //dlsym
#include <sched.h>  //getcpu
#include <stdio.h>
#include <stdlib.h> //aligned_alloc
#include <unistd.h> 
#include <execinfo.h>

//
#define LEVEL0 0
#define LEVEL1 1

//Green
#define LIBTRACE_HEADER_OK  "\033[0;32m [#LIBTRACE#] :\033[0m"

//Red
#define LIBTRACE_HEADER_NOK  "\033[0;31m [#LIBTRACE#] :\033[0m"

//
#define CAPTURE_LIMIT 100

//
#define MAX_TIMESTAMP 256

//
#define STACKTRACE_LINES 20

/*
 * Enumerated and derived types from cblas.h
 */
#ifdef WeirdNEC
   #define CBLAS_INDEX long
#else
   #define CBLAS_INDEX int
#endif

typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;

typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */

//GEMM call capture entry 
typedef struct gemm_call_s {

  //Call timestamp
  time_t call_timestamp;

  //Process ID
  pid_t pid;

  //Thread ID
  pid_t tid;

  //NUMA node
  unsigned node;

  //Core
  unsigned core;

  //Function parameters
  CBLAS_LAYOUT layout;
  CBLAS_TRANSPOSE TransA;
  CBLAS_TRANSPOSE TransB;
  CBLAS_INDEX M;
  CBLAS_INDEX N;
  CBLAS_INDEX K;
  double alpha;
  const double *A;
  CBLAS_INDEX lda;
  const double *B;
  CBLAS_INDEX ldb;
  double beta;
  const double *C;
  CBLAS_INDEX ldc;

  //Function pointer (tracer hook)
  void *ptr;

  //Original pointer (target function)
  void *optr;

  //Backtrace
  unsigned long long bt_size;
  void *bt_array[STACKTRACE_LINES + 2];
  
} gemm_call_t;

//
unsigned char limit_reached = LEVEL0;

//Captured entries/calls table
gemm_call_t *calls = NULL;

//Number of captured entries/call
unsigned long long nb_calls = 0;

//
static const char *target_functions[] = { "cblas_dgemm", NULL };

//Library constructor - called ahen the library is loaded
void libtrace_initialize() __attribute__((constructor));

//Library destructor - called when the library is unloaded 
void libtrace_finalize() __attribute__((destructor));

//Register the function dignature
typedef void (*real_cblas_dgemm_t)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
				   CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
				   const CBLAS_INDEX K, const double alpha, const double *A,
				   const CBLAS_INDEX lda, const double *B, const CBLAS_INDEX ldb,
				   const double beta, double *C, const CBLAS_INDEX ldc);

//
//The real CBLAS routine
void real_cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
		      CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
		      const CBLAS_INDEX K, const double alpha, const double *A,
		      const CBLAS_INDEX lda, const double *B, const CBLAS_INDEX ldb,
		      const double beta, double *C, const CBLAS_INDEX ldc)
{
  //Use dlsym to capture the real function's pointer address
  real_cblas_dgemm_t fptr = dlsym(RTLD_NEXT, target_functions[0]);

  //Call the real function
  fptr(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

//CBLAS routine tracer
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const CBLAS_INDEX M, const CBLAS_INDEX N,
                 const CBLAS_INDEX K, const double alpha, const double *A,
                 const CBLAS_INDEX lda, const double *B, const CBLAS_INDEX ldb,
                 const double beta, double *C, const CBLAS_INDEX ldc)
{
  //Obviously (memory is not infinite)!
  if (nb_calls < CAPTURE_LIMIT)
    {
      //Get timestamp
      calls[nb_calls].call_timestamp = time(NULL);

      //Get process ID
      calls[nb_calls].pid = getpid();
      
      //Get thread ID
      calls[nb_calls].tid = gettid();

      //Get core and NUMA node
      getcpu(&calls[nb_calls].core, &calls[nb_calls].node);
      
      //Record call parameters
      calls[nb_calls].layout = layout;
      calls[nb_calls].TransA = TransA;
      calls[nb_calls].TransB = TransB;
      calls[nb_calls].M = M;
      calls[nb_calls].N = N;
      calls[nb_calls].K = K;
      calls[nb_calls].alpha = alpha;
      calls[nb_calls].A = A;
      calls[nb_calls].lda = lda;
      calls[nb_calls].B = B;
      calls[nb_calls].ldb = ldb;
      calls[nb_calls].beta = beta;
      calls[nb_calls].C = C;
      calls[nb_calls].ldc = ldc;

      //Save function pointer address
      calls[nb_calls].ptr = (void *)cblas_dgemm;

      //Get the backtrace
      calls[nb_calls].bt_size = backtrace(calls[nb_calls].bt_array, STACKTRACE_LINES + 2);
      
      //
      nb_calls++;      
    }
  else
    {
      //Print the message only once!
      if (limit_reached == LEVEL0)
	{
	  printf(" %s capture limit is reached, waiting for the program to finish before dumping\n", LIBTRACE_HEADER_NOK);
	  limit_reached = LEVEL1;
	}
    }
  
  //Call the real function
  real_cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

//
void libtrace_dump()
{
  //Dump only if calls captured
  if (nb_calls)
    {
      FILE *fp = fopen("libtrace.out", "wb");
      
      if (!fp)
	{
	  printf(" %s cannot create output file!\n", LIBTRACE_HEADER_NOK);
	  exit(1);
	}

      fprintf(fp, " %s captured %llu calls to %s\n", LIBTRACE_HEADER_OK, nb_calls, target_functions[0]);
      
      //
      char timestamp[MAX_TIMESTAMP];
      
      //
      for (unsigned long long i = 0; i < nb_calls; i++)
	{
	  struct tm t = *localtime(&calls[i].call_timestamp);
	  
	  sprintf(timestamp, "%d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
	  
	  fprintf(fp, "\ncall #%llu (%s):\n"
		  
		  "\tPID: %u\n"
		  "\tTID: %u\n"
		  
		  "\tNUMA node: %u\n"
		  "\tCPU core : %u\n"
		  
		  "\tlayout: %u\n"  //layout
		  "\tTransA: %u\n"  //TransA
		  "\tTransB: %u\n"  //TransB
		  "\tM     : %d\n" //M
		  "\tN     : %d\n" //N
		  "\tK     : %d\n" //K
		  "\talpha : %lf\n" //alpha
		  "\tA(ptr): %p\n"  //A
		  "\tlda   : %d\n" //lda
		  "\tB(ptr): %p\n"  //B
		  "\tldb   : %d\n" //ldb
		  "\tbeta  : %lf\n" //beta
		  "\tC(ptr): %p\n"  //C
		  "\tldc   : %d\n" //ldc

		  "\tptr   : %p\n"
		  ,
		  i,
		  timestamp,
		  
		  calls[i].pid,
		  calls[i].tid,
		  
		  calls[i].node,
		  calls[i].core,
		  
		  calls[i].layout,
		  calls[i].TransA,
		  calls[i].TransB,
		  calls[i].M,
		  calls[i].N,
		  calls[i].K,
		  calls[i].alpha,
		  calls[i].A,
		  calls[i].lda,
		  calls[i].B,
		  calls[i].ldb,
		  calls[i].beta,
		  calls[i].C,
		  calls[i].ldc,
		  calls[i].ptr);

	  if (calls[i].bt_array)
	    {
	      fprintf(fp, "\tbacktrace:\n");
	      
	      char **bt_symbols = backtrace_symbols(calls[i].bt_array + 2, calls[i].bt_size - 2);
	      
	      if (bt_symbols)
		{
		  for (unsigned long long j = 0; j < calls[i].bt_size - 2; j++)
		    fprintf(fp, "\t\t%s\n", bt_symbols[j]);

		  fprintf(fp, "\n\n");

		  //
		  free(bt_symbols);
		}
	      else
		fprintf(fp, "\tbacktrace empty\n");
	    }
	}
      
      //
      fclose(fp);
    }
  else
    printf(" %s no calls captured\n", LIBTRACE_HEADER_NOK);
}

//
void libtrace_initialize()
{
  //Green
  
  printf(" %s initialized ...\n", LIBTRACE_HEADER_OK);
  
  //
  nb_calls = 0;

  //Allocate 
  calls = malloc(sizeof(gemm_call_t) * CAPTURE_LIMIT);
}

//
void libtrace_finalize()
{
  //
  libtrace_dump();
  
  //
  free(calls);
  
  //
  printf("\n %s finilized\n", LIBTRACE_HEADER_OK);
}
