#include <iostream>
#include <chrono>
#include <time.h>
#include <algorithm>

#define eChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int N_POINTS = 5, DIM_SIZE = 3;

void runAndTime(void (*f)());
void generatePoints(int3 *points, int n);
void buildKDTree(int3 *points, int3 *tree, int n, int m);
void cpu(int3 *points, int3 *tree, int n, int m);
void gpu(int3 *points, int3 *tree, int n, int m);


int main() {
    srand(16);

    int TREE_SIZE = 1;
    while(TREE_SIZE < N_POINTS) TREE_SIZE <<= 1;

    int3 *points;
    int3 *tree;
    eChk(cudaMallocManaged(&points, N_POINTS * sizeof(int3)));

    generatePoints(points, N_POINTS);
    buildKDTree(points, tree, N_POINTS, TREE_SIZE);

    runAndTime([&]() -> void { cpu(points, tree); });
    runAndTime([&]() -> void { gpu(points, tree); });

    eChk(cudaFree(points));
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


void buildSubTree(int3 *points, int3 *tree, int start, int end, int depth, int node) {
    if(start == end) {
        tree[node] = points[start];
        return;
    }

    std::sort(points+start, points+end, [depth](int3 p1, int3 p2) -> bool {
        if(depth % 3 == 0) return p1.x < p2.x;
        if(depth % 3 == 1) return p1.y < p2.y;
        return p1.z < p2.z;
    });

    int split = (start + end)/2;

    tree[node] = points[split];

    buildSubTree(points, tree, start, split, depth+1, node*2);
    buildSubTree(points, tree, split+1, end, depth+1, node*2 + 1);
}

void buildKdTree(int3 *points, int3 *tree, int n, int m) {
    buildSubTree(points, tree, 0, n, 0, 1);
}

void print(int3 *points, int n) {
    for(int i = 0; i < n; i++) {
        std::cout<<"["<<points[i].x<<points[i].y<<points[i].z<<"] ";
    }
    std::cout<<std::endl;
}

void cpu(int3 *points, int3 *tree, int n, int m) {
    print(points, n);
    print(points, m);
}

void gpu(int3 *points, int3 *tree, int n, int m)
{
    
}
