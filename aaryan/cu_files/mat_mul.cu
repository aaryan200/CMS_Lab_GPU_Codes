#include<iostream>
#include<math.h>
#include<chrono>
#include<iomanip>
using namespace std;
using namespace chrono;

void printMatrix(float* x, int a, int b) {
    for (int i=0; i < a; i++) {
        for (int j=0; j < b; j++) {
            cout << x[b*i + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

float random_float_32() {
    // Generate a random 32-bit integer
    uint32_t random_int = rand();

    // Convert the random integer to a floating-point number
    float random_float = *((float*)&random_int);

    return random_float;
}

__device__
float mult_row_col(float *p, float *q, int a, int b, int c, int row, int col) {
    // Multiply row of p with col of q
    float res = 0.0f;

    // Starting index of P matrix and Q matrix
    int p_ind = row * b, q_ind = col;

    for (int i=0; i < b; i++) {
        res += p[p_ind]*q[q_ind];

        p_ind += 1;
        q_ind += c;
    }

    return res;
}

__global__
void matmul(float *p, float *q, float* r, int a, int b, int c) {
    int numBlocks = gridDim.x;
    int numThrPerBlock = blockDim.x;
    int blockId = blockIdx.x, threadId = threadIdx.x;

    int index = blockId * numThrPerBlock + threadId;
    int stride = numBlocks * numThrPerBlock; // Total number of threads

    int numOps = a*c;

    int row, col;

    for (int i=index; i < numOps; i += stride) {
        row = i / c;
        col = i - row * c;

        r[row*a + col] = mult_row_col(p, q, a, b, c, row, col);
    }

    return;
}

int main() {
    int a, b, c;
    cout << "Enter a, b, c: ";
    cin >> a >> b >> c;

    int sz_p = a*b, sz_q = b*c, sz_r = a*c;

    // Create matrices P, Q and R of size [a*b], [b*c] and [a*c] respectively
    float *p, *q, *r;
    cudaMallocManaged(&p, sz_p*sizeof(float));
    cudaMallocManaged(&q, sz_q*sizeof(float));
    cudaMallocManaged(&r, sz_r*sizeof(float));

    for (int i=0; i < sz_p; i++) {
        //p[i] = random_float_32();
	p[i] = 1.0f;
    }

    for (int i=0; i < sz_q; i++) {
        //q[i] = random_float_32();
	q[i] = 2.0f;
    }

//    cout << "P:"<<endl;
    //printMatrix(p, a, b);
//    cout << "Q:"<<endl;
    //printMatrix(q, b, c);

    int numBlocks, numThrPerBlock;
    cout << "Enter number of blocks: ";
    cin >> numBlocks;

    cout << "Enter number of threads per block: ";
    cin >> numThrPerBlock;

    auto start = high_resolution_clock::now();

    matmul<<<numBlocks, numThrPerBlock>>>(p, q, r, a, b, c);

    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();

    double timeTaken = duration_cast<milliseconds>(end - start).count();

    cout << fixed << setprecision(4);

    cout << "Time taken: " << timeTaken << "ms" << endl;

//    cout << "R:"<<endl;
    //printMatrix(r, a, c);

    cudaFree(p);
    cudaFree(q);
    cudaFree(r);

    return 0;
}
