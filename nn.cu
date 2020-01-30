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

const int N_POINTS = 5, N_QUERIES = 5, INF = 1e9;

void runAndTime(void (*f)(int3*, int, int3*, int), int3 *tree, int treeSize, int3 *queries, int nQueries);
void print(int3 *points, int n);
void generatePoints(int3 *points, int n);
void buildKDTree(int3 *points, int3 *tree, int n, int m);
void cpu(int3 *tree, int treeSize, int3 *queries, int nQueries);
void gpu(int3 *tree, int treeSize, int3 *queries, int nQueries);


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

    print(points, N_POINTS);

    runAndTime(cpu, tree, TREE_SIZE, queries, N_QUERIES);
    runAndTime(gpu, tree, TREE_SIZE, queries, N_QUERIES);

    eChk(cudaFree(points));
    eChk(cudaFree(tree));
    eChk(cudaFree(queries));
}

void runAndTime(void (*f)(int3*, int, int3*, int), int3 *tree, int treeSize, int3 *queries, int nQueries)
{
    auto start = std::chrono::system_clock::now();
    f(tree, treeSize, queries, nQueries);
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
    if(start >= end) {
        return;
    }

    std::sort(points+start, points+end, [depth](int3 p1, int3 p2) -> bool {
        if(depth % 3 == 0) return p1.x < p2.x;
        if(depth % 3 == 1) return p1.y < p2.y;
        return p1.z < p2.z;
    });

    int split = (start + end-1)/2;

    tree[node].x = points[split].x;
    tree[node].y = points[split].y;
    tree[node].z = points[split].z;

    print(points+start, end-start);
    print(points+split, 1);

    buildSubTree(points, tree, start, split, depth+1, node*2);
    buildSubTree(points, tree, split+1, end, depth+1, node*2 + 1);
}

void buildKDTree(int3 *points, int3 *tree, int n, int treeSize) {
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

int3 findNearestNeighbor(int3 *tree, int treeSize, int treeNode, int depth, int3 query) {
    int val1, val2;
    if(depth % 3 == 0) {
        val1 = tree[treeNode].x;
        val2 = query.x;
    } else if(
        val1 = tree[treeNode].y;
        val2 = query.y;
    ) else {
        val1 = tree[treeNode].z;
        val2 = query.z;
    }

    if(val1 < val2) {
        if(treeNode*2 < treeSize && tree[treeSize*2] != make_int3(-INF, -INF, -INF)) {
            return findNearestNeighbor(tree, treeSize, treeNode*2, query);
        }
    } else if(val1 > val2) {
        if(treeNode*2+1 < treeSize && tree[treeSize*2+1] != make_int3(-INF, -INF, -INF)) {
            return findNearestNeighbor(tree, treeSize, treeNode*2+1, query);
        }
    }
    return tree[treeNode];
}

void cpu(int3 *tree, int treeSize, int3 *queries, int nQueries) {
    int3 *results = new int3[nQueries];

    for(int i = 0; i < nQueries; i++) {
        results[i] = findNearestNeighbor(tree, treeSize, 1, 0, queries[i]);
    }

    print(queries, nQueries);
    print(results, nQueries);
}

void gpu(int3 *tree, int treeSize, int3 *queries, int nQueries)
{
    
}
