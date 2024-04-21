// Compile this file using:
// gcc 03_fftw_fft2.c -o fftw_fft2 -lfftw3 -lgsl -lgslcblas -lm
// Run the executable using:
// ./executable Nx Ny
// Nx and Ny are the dimensions of the 2D grid
// Example:
// ./fftw_fft2 100 100
#include<stdio.h>
#include<stdbool.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<fftw3.h>
#include<sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

int main(int argc, char* argv[]) {
    int Nx, Ny;
    Nx = atoi(argv[1]);
    Ny = atoi(argv[2]);

    int arr_size = Nx * Ny;

    fftw_complex *c, *cnew;

    c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);
    cnew = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);

    struct timeval start, end;

    gettimeofday(&start, NULL);

    fftw_plan plan_forward_c;

    plan_forward_c = fftw_plan_dft_2d(Nx, Ny, c, cnew, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan_forward_c);

    gettimeofday(&end, NULL);

    long long seconds, milliseconds;

    seconds = end.tv_sec - start.tv_sec;
    milliseconds = (end.tv_usec - start.tv_usec)/1000;

    if (milliseconds < 0) {
        seconds -= 1;
        milliseconds += 1000;
    }

    double time_taken = seconds*1.0 + milliseconds/1000.0;

    printf("%.9f", time_taken);

    fftw_destroy_plan(plan_forward_c);
    fftw_free(c);
    fftw_free(cnew);
    fftw_cleanup();

    return 0;
}