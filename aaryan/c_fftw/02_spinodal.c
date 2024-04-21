// Compile this file using:
// gcc 02_spinodal.c -o spinodal_c -lfftw3 -lgsl -lgslcblas -lm
// Run the executable using:
// ./executable Nx Ny num_iter --v
// Nx and Ny are the dimensions of the 2D grid
// num_iter is the number of iterations to run the simulation
// --v is an optional argument to print the progress of the simulation
// Example:
// ./spinodal_c 100 100 10
#include<stdio.h>
#include<stdbool.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<fftw3.h>
#include<sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#define PI 3.14159265358979323846

double spinodal_2d(int Nx, int Ny, double dx, double dy,
        double dt, double A, double M, double kappa,
        fftw_complex* c, fftw_complex* cnew, fftw_complex* chat,
        fftw_complex* g, fftw_complex* ghat,
        int num_iter, int steps_per_iter,
        bool verbose) 
{
    int m, n, i, j, idx;
    double temp_val;
    double delkx = 2.0*PI/(Nx*dx);
    double delky = 2.0*PI/(Ny*dy);

    double kx, ky, k2, k4;

    double nx_half = Nx/2.0, ny_half = Ny/2.0;

    int arr_size = Nx * Ny;

    for (i = 0; i < arr_size; i++) {
        cnew[i][0] = c[i][0];
        cnew[i][1] = c[i][1];
    }

    struct timeval start, end;

    gettimeofday(&start, NULL);

    for (m = 0; m < num_iter; m++) {
        if (verbose) {
            printf("Iteration %d: [", m+1);
        }
        for (n = 0; n < steps_per_iter; n++) {
            // First derivative
            // g = df/dc = 2*A*cnew*(1-cnew)*(1-2cnew)
            for (i = 0; i < arr_size; i++) {
                temp_val = cnew[i][0];
                g[i][0] = 2.0*A*temp_val*(1.0-temp_val)*(1.0-2.0*temp_val);
                g[i][1] = 0.0;
            }
            // Declare plan variables
            fftw_plan plan_forward_g, plan_forward_c;
            fftw_plan plan_backward_c;

            // ghat = fft2(g)
            plan_forward_g = fftw_plan_dft_2d(Nx, Ny, g, ghat, FFTW_FORWARD, FFTW_ESTIMATE);

            fftw_execute(plan_forward_g);

            // chat = fft2(cnew)
            plan_forward_c = fftw_plan_dft_2d(Nx, Ny, cnew, chat, FFTW_FORWARD, FFTW_ESTIMATE);

            fftw_execute(plan_forward_c);
            
            for (i = 0; i < Nx; i++) {
                if (i <= nx_half) {
                    kx = i*delkx;
                } else {
                    kx = (i-Nx)*delkx;
                }
                for (j = 0; j < Ny; j++) {
                    if (j <= ny_half) {
                        ky = j*delky;
                    } else {
                        ky = (j-Ny)*delky;
                    }
                    k2 = kx*kx + ky*ky;
                    k4 = k2*k2;

                    temp_val = 1+2.0*M*kappa*k4*dt;

                    idx = Ny*i + j;

                    // Real part
                    chat[idx][0] = (chat[idx][0] - M*dt*k2*ghat[idx][0])/temp_val;

                    // Imaginary part
                    chat[idx][1] = (chat[idx][1] - M*dt*k2*ghat[idx][1])/temp_val;
                }
            }

            // cnew = ifft2(chat).real
            plan_backward_c = fftw_plan_dft_2d(Nx, Ny, chat, cnew, FFTW_BACKWARD, FFTW_ESTIMATE);

            fftw_execute(plan_backward_c);

            for (i = 0; i < arr_size; i++) {
                // Need to divide by Nx*Ny to get the correct values
                cnew[i][0] = cnew[i][0]/(1.0*arr_size);
                c[i][0] = cnew[i][0];

                cnew[i][1] = 0.0;
                c[i][1] = cnew[i][1];
            }

            fftw_destroy_plan(plan_forward_g);
            fftw_destroy_plan(plan_forward_c);
            fftw_destroy_plan(plan_backward_c);

            // Deallocate memory associated with FFTW plans
            fftw_cleanup();

            if (verbose) {
                printf("=");
                fflush(stdout);
            }
        }
        if (verbose) {
            printf("]\n");
        }
    }
    if (verbose) printf("\n");
    
    gettimeofday(&end, NULL);

    long long seconds, milliseconds;

    seconds = end.tv_sec - start.tv_sec;
    milliseconds = (end.tv_usec - start.tv_usec)/1000;

    if (milliseconds < 0) {
        seconds -= 1;
        milliseconds += 1000;
    }

    double time_taken = seconds*1.0 + milliseconds/1000.0;
    return time_taken;
}

int main(int argc, char* argv[]) {
    bool verbose;
    int Nx, Ny, num_iter;

    if (argc < 4) {
        printf("Please provide the dimensions of the 2D grid and the number of iterations\n");
        return 1;
    }

    Nx = atoi(argv[1]);
    Ny = atoi(argv[2]);
    num_iter = atoi(argv[3]);

    if (argc > 4) {
        // Check if verbose flag is set
        if (strcmp(argv[4], "--v") == 0) {
            verbose = true;
        } else {
            verbose = false;
        }
    } else {
        verbose = false;
    }

    int i;
    double dx = 1.0, dy = 1.0;
    double dt = 0.5, A = 1.0, M = 1.0, kappa = 1.0;
    fftw_complex *c, *cnew, *chat, *g, *ghat;
    int steps_per_iter = 100;
    int arr_size = Nx * Ny;

    c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);
    cnew = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);
    chat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);
    g = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);
    ghat = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*arr_size);

    const gsl_rng_type *rng_type;
    gsl_rng *rng;

    // Create a random number generator
    gsl_rng_env_setup();
    rng_type = gsl_rng_default;
    rng = gsl_rng_alloc(rng_type);

    // Set seed for reproducibility
    gsl_rng_set(rng, 1024);

    for (i = 0; i < Nx*Ny; i++) {
        double rand_noise = gsl_ran_gaussian(rng, 0.01); // Mean = 0, Standard Deviation = 0.01
        c[i][0] = 0.5 - rand_noise;
        c[i][1] = 0.0;
    }

    double time_taken = spinodal_2d(Nx, Ny, dx, dy, dt, A, M, kappa,
            c, cnew, chat, g, ghat, num_iter, steps_per_iter, verbose);

    // printf("Time taken: %f seconds\n", time_taken);
    printf("%.9f", time_taken);

    fftw_free(c);
    fftw_free(cnew);
    fftw_free(chat);
    fftw_free(g);
    fftw_free(ghat);

    gsl_rng_free(rng);

    return 0;
}