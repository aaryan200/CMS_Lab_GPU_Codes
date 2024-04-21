#include <iostream>
#include <math.h>
using namespace std;

int N = 1 << 20;

__global__
void test_() {
    // Won't work because a host variable cannot be accessed directly into a device function
    printf("%d\n", N);
    return;
}

int main() {

    // float *x, *y;

    // cudaMallocManaged(&x, N*sizeof(float));
    // cudaMallocManaged(&y, N*sizeof(float));

    // for (int i=0;i<N;i++) {
    //     x[i] = 1.0f;
    //     y[i] = 2.0f;
    // }

    test_<<<2,256>>>();

    cudaDeviceSynchronize();

    // float maxErr = 0.0f;

    // for (int i=0;i<N;i++) {
    //     maxErr = fmax(maxErr, fabs(y[i] - 3.0f));
    // }

    // cout << "Max error is: " << maxErr << endl;

    // cudaFree(x);
    // cudaFree(y);

    return 0;
}