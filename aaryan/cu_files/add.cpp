#include <iostream>
#include <math.h>
using namespace std;

void add(int n, float* x, float* y) {
    for (int i=0;i<n;i++) {
        y[i] = x[i] + y[i];
    }
    return;
}

int main() {
    int N = 1 << 20;

    float* x = new float[N];
    float* y = new float[N];

    for (int i=0;i<N;i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add(N, x, y);

    float maxErr = 0.0f;

    for (int i=0;i<N;i++) {
        maxErr = fmax(maxErr, fabs(y[i] - 3.0f));
    }

    cout << "Max error is: " << maxErr << endl;

    delete[] x;
    delete[] y;

    return 0;
}