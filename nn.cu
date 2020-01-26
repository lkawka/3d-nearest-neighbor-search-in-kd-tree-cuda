#include <iostream>
#include <chrono>
#include <time.h>

#define eChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int N = 5, DIM_SIZE = 3;

typedef struct __align__(16) {
    int3 value;
    int splitDim;
} KDNode;

void runAndTime(void (*f)());
void cpu();
void gpu();


int main() {
    srand(16);

    runAndTime(cpu);
    runAndTime(gpu);
}

void runAndTime(void (*f)())
{
    auto start = std::chrono::system_clock::now();
    f();
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();
    std::cout << "Elapsed time in milliseconds : " << duration << "ms\n\n";
}

void generatePoints(int3 *points, int n) {
    for(int i = 0; i < n; i++) {
        points[i] = make_int3(rand()%100, rand()%100, rand()%100);
    }
}

void buildKdTree(int3 *points, KDNode *tree, int n) {
    for(int i = 0; i < n; i++) {
        tree[i] = { .value = points[i] };
    }
}

void cpu() {
    int3 *points = new int[N]];
    KDNode *tree = new KDNode[N];

    generatePoints(points, N);
    buildKdTree(points, tree, N);
}

void gpu()
{
    int3 *points;
    KDNode *tree;

    eChk(cudaMallocManaged(&points, N * sizeof(int3)));
    eChk(cudaMallocManaged(&tree, N * sizeof(KDNode)));

    generatePoints(points, N);
    buildKdTree(points, tree, N);

    eChk(cudaFree(points));
}
