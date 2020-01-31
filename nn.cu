#include <iostream>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <math.h>

#define eChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int N_POINTS = 1e3, N_QUERIES = 1e6, INF = 1e9, RANGE_MAX = 100, N_PRINT = 10;

void print(int3 *points, int n);
__host__ void generatePoints(int3 *points, int n);
__host__ void buildKDTree(int3 *points, int3 *tree, int n, int m);
__global__ void nearestNeighborGPU(int3 *tree, int treeSize, int3 *queries, int3 *results, int nQueries);
__host__ void printResults(int3 *queries, int3 *results, int n);

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

    auto start = std::chrono::system_clock::now();

    int3 *results;
    eChk(cudaMallocManaged(&results, N_QUERIES * sizeof(int3)));

    nearestNeighborGPU<<<32768, 8>>>(tree, TREE_SIZE, queries, results, N_QUERIES);
    eChk(cudaDeviceSynchronize());
    
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();

    printResults(queries, results, N_PRINT);

    std::cout << "Elapsed time in milliseconds : " << duration << "ms\n\n";

    eChk(cudaFree(results));
    eChk(cudaFree(points));
    eChk(cudaFree(tree));
    eChk(cudaFree(queries));
}

__host__ void generatePoints(int3 *points, int n) {
    for(int i = 0; i < n; i++) {
        points[i] = make_int3(rand()%RANGE_MAX, rand()%RANGE_MAX, rand()%RANGE_MAX);
    }
}

__host__ void buildSubTree(int3 *points, int3 *tree, int start, int end, int depth, int node) {
    if(start >= end) return;

    std::sort(points + start, points + end, [depth](int3 p1, int3 p2) -> bool {
        if(depth % 3 == 0) return p1.x < p2.x;
        if(depth % 3 == 1) return p1.y < p2.y;
        return p1.z < p2.z;
    });

    int split = (start + end - 1)/2;
    tree[node] = points[split];

    buildSubTree(points, tree, start, split, depth+1, node*2);
    buildSubTree(points, tree, split + 1, end, depth+1, node*2 + 1);
}

__host__ void buildKDTree(int3 *points, int3 *tree, int n, int treeSize) {
    for(int i = 0; i < treeSize; i++) {
        tree[i] = make_int3(-INF, -INF, -INF);
    }

    buildSubTree(points, tree, 0, n, 0, 1);
}

void print(int3 *points, int n) {
    for(int i = 0; i < n; i++) {
        std::cout<<"["<<points[i].x<<", "<<points[i].y<<", "<<points[i].z<<"] ";
    }
    std::cout<<std::endl;
}

__device__ int3 getCloser(int3 p, int3 p2, int3 p3)
{
    if ((pow(p.x - p2.x, 2) + pow(p.y - p2.y, 2) + pow(p.z - p2.z, 2)) < (pow(p.x - p3.x, 2) + pow(p.y - p3.y, 2) + pow(p.z - p3.z, 2)))
    {
        return p2;
    }
    return p3;
}

__device__ int3 findNearestNeighbor(int3 *tree, int treeSize, int treeNode, int depth, int3 query) 
{
    int3 node = tree[treeNode];

    int val1, val2;
    if (depth % 3 == 0)
    {
        val1 = node.x;
        val2 = query.x;
    }
    else if (depth % 3 == 1)
    {
        val1 = node.y;
        val2 = query.y;
    }
    else
    {
        val1 = node.z;
        val2 = query.z;
    }

    if ((val1 < val2) && (treeNode * 2 < treeSize))
    {
        int3 leftChild = tree[treeNode * 2];
        if (leftChild.x != -INF && leftChild.y != -INF && leftChild.z != -INF)
        {
            return getCloser(query, node, findNearestNeighbor(tree, treeSize, treeNode * 2, depth + 1, query));
        }
    }
    else if ((val1 > val2) && (treeNode * 2 + 1 < treeSize))
    {
        int3 rightChild = tree[treeNode * 2 + 1];
        if (rightChild.x != -INF && rightChild.y != -INF && rightChild.z != -INF)
        {
            return getCloser(query, node, findNearestNeighbor(tree, treeSize, treeNode * 2 + 1, depth + 1, query));
        }
    }
    return node;
}

__global__ void nearestNeighborGPU(int3 *tree, int treeSize, int3 *queries, int3 *results, int nQueries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < nQueries) {
        results[index] = findNearestNeighbor(tree, treeSize, 1, 0, queries[index]);
    }
}

__host__ void printResults(int3 *queries, int3 *results, int n) {
    for(int i = 0; i < n; i++) {
        std::cout<<"query: ["<<queries[i].x<<", "<<queries[i].y<<", "<<queries[i].z<<"] ";
        std::cout<<", result: ["<<results[i].x<<", "<<results[i].y<<", "<<results[i].z<<"] ";
        std::cout<<", distance: "<<sqrt(pow(queries[i].x - results[i].x, 2) + pow(queries[i].y - results[i].y, 2) + pow(queries[i].z - results[i].z, 2));
        std::cout<<std::endl;
    }
}