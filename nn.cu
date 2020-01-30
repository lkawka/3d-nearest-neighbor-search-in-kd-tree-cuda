#include <iostream>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <limits.h>

#define eChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int N_POINTS = 5, N_QUERIES = 5;

void runAndTime(void (*f)(int3*, int, int3*, int), int3 *tree, int tree_size, int3 *queries, int nQueries);
void print(int3 *points, int n);
void generatePoints(int3 *points, int n);
void buildKDTree(int3 *points, int3 *tree, int n, int m);
void cpu(int3 *tree, int tree_size, int3 *queries, int nQueries);
void gpu(int3 *tree, int tree_size, int3 *queries, int nQueries);


int main() {
    srand(16);

    int TREE_SIZE = 1;
    while(TREE_SIZE < N_POINTS) TREE_SIZE <<= 1;

    int3 *points;
    int3 *tree;
    int3 *queries;
    eChk(cudaMallocManaged(&points, N_POINTS * sizeof(int3)));
    eChk(cudaMallocManaged(&tree, TREE_SIZE * sizeof(int3)));
    eChk(cudaMallocManaged(&queries, N_QUERIES * sizeof(int3)));

    generatePoints(points, N_POINTS);
    buildKDTree(points, tree, N_POINTS, TREE_SIZE);
    generatePoints(queries, N_QUERIES);

    runAndTime(cpu, tree, TREE_SIZE, queries, N_QUERIES);
    runAndTime(gpu, tree, TREE_SIZE, queries, N_QUERIES);

    eChk(cudaFree(points));
    eChk(cudaFree(tree));
    eChk(cudaFree(queries));
}

void runAndTime(void (*f)(int3*, int, int3*, int), int3 *tree, int tree_size, int3 *queries, int nQueries)
{
    auto start = std::chrono::system_clock::now();
    f(tree, tree_size, queries, nQueries);
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

void buildKDTree(int3 *points, int3 *tree, int n, int tree_size) {
    for(int i = 0; i < tree_size, i++) {
        tree[i] = make_int3(MIN_INT, MIN_INT, MIN_INT);
    }

    buildSubTree(points, tree, 0, n, 0, 1);
}

void print(int3 *points, int n) {
    for(int i = 0; i < n; i++) {
        std::cout<<"["<<points[i].x<<", "<<points[i].y<<", "<<points[i].z<<"] ";
    }
    std::cout<<std::endl;
}

void cpu(int3 *tree, int tree_size, int3 *queries, int nQueries) {
    print(tree+1, tree_size);
    print(queries, nQueries);
}

void gpu(int3 *tree, int tree_size, int3 *queries, int nQueries)
{
    
}
