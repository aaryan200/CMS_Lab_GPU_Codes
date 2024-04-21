// To compile run:
// gcc 01_fftw_test.c -lfftw3 -lm
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <fftw3.h>


int main() {
    fftw_complex* in;
    fftw_complex *out;
    fftw_plan plan;
    int n0 = 3, n1 = 3;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n0*n1);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n0*n1);

    // Initialize in
    for (int i=0;i<n0;i++) {
        for (int j=0;j<n1;j++) {
            in[i*n1+j][0] = 0.2*i + 0.3*j;
            in[i*n1+j][1] = 0.0;
        }
    }

    // Print in
    printf("Input array: \n");
    for (int i=0;i<n0;i++) {
        for (int j=0;j<n1;j++) {
            printf("%f +%fi, ", in[i*n1+j][0], in[i*n1+j][1]);
        }
        printf("\n");
    }

    // FFT of in
    plan = fftw_plan_dft_2d(n0, n1, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    // Print the result
    printf("Output array: \n");
    for (int i=0;i<n0;i++) {
        for (int j=0;j<n1;j++) {
            printf("%f +%fi, ", out[i*n1+j][0], out[i*n1+j][1]);
        }
        printf("\n");
    }

    // Inverse FFT of out
    plan = fftw_plan_dft_2d(n0, n1, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    // Print the result
    printf("Inverse FFT array: \n");
    for (int i=0;i<n0;i++) {
        for (int j=0;j<n1;j++) {
            printf("%f +%fi, ", in[i*n1+j][0]/(n0*n1), in[i*n1+j][1]/(n0*n1));
        }
        printf("\n");
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return 0;
}