#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s NX NY\n", argv[0]);
        return 1;
    }

    // Read the matrix size from the command line arguments
    int NX = atoi(argv[1]);
    int NY = atoi(argv[2]);

    // Allocate memory for the data on the host
    cufftComplex *data = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
    if (data == NULL) {
        printf("Error allocating memory for data.\n");
        return 1;
    }

    // Randomly initialize the data
    srand(time(NULL)); // Seed the random number generator
    for (int i = 0; i < NX * NY; ++i) {
        data[i].x = (float)rand() / RAND_MAX;
        data[i].y = (float)rand() / RAND_MAX;
    }

    // Allocate memory for the data on the device
    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex) * NX * NY);
    
    // Copy data from host to device
    cudaMemcpy(d_data, data, sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice);

    // Create a 2D FFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, NX, NY, CUFFT_C2C);

    // Measure the start time
    clock_t start = clock();

    // Execute the FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Wait for completion
    cudaDeviceSynchronize();

    // Measure the end time
    clock_t end = clock();

    // Calculate the elapsed time
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_spent);

    // Copy the result back to the host
    cudaMemcpy(data, d_data, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost);

    // Process or use the transformed data here

    // Destroy the FFT plan
    cufftDestroy(plan);

    // Free memory on the device and host
    cudaFree(d_data);
    free(data);

    return 0;
}

