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

const int N = 5;

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

void cpu() {

}

void gpu()
{
    int3 *points;

    eChk(cudaMallocManaged(&points, N * sizeof(int3)));

    eChk(cudaFree(points));
}
