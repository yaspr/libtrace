#if defined (__INTEL_COMPILER)

#include <mkl.h>

#elif defined (__GNUC__) || (__clang__)

#include <cblas.h>

#endif

#include <stdio.h>
#include <stdlib.h>

#define ALIGN32 32

void init(double *restrict a, double c, unsigned long long n)
{
  for (unsigned long long i = 0; i < n; i++)
    a[i] = c;
}

void test_gemm(double *a, double *b, double *c, unsigned long long n, unsigned long long r)
{
  for (unsigned long long i = 0; i < r; i++)
    {
      cblas_dgemm(CblasRowMajor,
		  CblasNoTrans,
		  CblasNoTrans, n, n, n, 1.0, a, n, b, n, 1.0, c, n);
    }
}

int main(int argc, char **argv)
{
  if (argc < 3)
    return printf("Usage: %s [n] [r]\n", argv[0]), 1;
  
  unsigned long long n = atoll(argv[1]);
  unsigned long long r = atoll(argv[2]);
    
  unsigned long long s = sizeof(double) * n * n;
  
  double *restrict a = aligned_alloc(ALIGN32, s);
  double *restrict b = aligned_alloc(ALIGN32, s);
  double *restrict c = aligned_alloc(ALIGN32, s);
  
  init(a, 2.0, n * n);
  init(b, 3.0, n * n);
  init(c, 0.0, n * n);

  for (unsigned long long i = 0; i < r; i++)
    {
      cblas_dgemm(CblasRowMajor,
		  CblasNoTrans,
		  CblasNoTrans, n, n, n, 1.0, a, n, b, n, 1.0, c, n);
    }

  for (unsigned long long i = 0; i < 10; i++)
    {
      for (unsigned long long j = 0; j < 10; j++)
	printf("%lf ", c[i * n + j]);
      
      printf("\n"); 
    }

  test_gemm(a, b, c, n, r);
  
  free(a);
  free(b);
  free(c);
  
  return 0;
}
