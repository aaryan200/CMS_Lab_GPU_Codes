#include <iostream>
#include <math.h>
using namespace std;

__global__
void add(int n, float* x, float* y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    // cout << "Index: " << index << ", Stride: " << stride << endl;
    printf("Index: %d, Stride: %d\n", index, stride);
    for (int i=index; i < n; i += stride) {
        y[i] = y[i] + x[i];
    }
    return;
}

int main() {
    int N = 1 << 20;

    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i=0;i<N;i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1,256>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxErr = 0.0f;

    for (int i=0;i<N;i++) {
        maxErr = fmax(maxErr, fabs(y[i] - 3.0f));
    }

    cout << "Max error is: " << maxErr << endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}