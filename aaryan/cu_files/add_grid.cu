#include<iostream>
#include<math.h>
using namespace std;

__global__
void add(int n, float *x, float *y) {
    int numBlocks = gridDim.x;
    int numThrPerBlock = blockDim.x;
    int blockId = blockIdx.x, threadId = threadIdx.x;

    int index = blockId * numThrPerBlock + threadId;
    int stride = numBlocks * numThrPerBlock; // Total number of threads

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

    for (int i=0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int numThrPerBlock = 256;
    // numBlocks should be N/numThrPerBlock
    // Take ceil
    int numBlocks = (N + numThrPerBlock - 1) / numThrPerBlock;

    add<<<numBlocks, numThrPerBlock>>>(N, x, y);

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