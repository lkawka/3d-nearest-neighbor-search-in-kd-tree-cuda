#include <iostream>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <math.h>

const int N_POINTS = 1e4, N_QUERIES = 1e6, INF = 1e9, RANGE_MAX = 100, N_PRINT = 10;

struct int3
{
    int x;
    int y;
    int z;
};

void print(int3 *points, int n);
void generatePoints(int3 *points, int n);
void buildKDTree(int3 *points, int3 *tree, int n, int m);
int3 findNearestNeighbor(int3 *tree, int treeSize, int treeNode, int depth, int3 query);
void printResults(int3 *queries, int3 *results, int n);

int main()
{
    srand(17);

    int TREE_SIZE = 1;
    while (TREE_SIZE < N_POINTS)
        TREE_SIZE <<= 1;

    int3 *points = new int3[N_POINTS];
    int3 *tree = new int3[TREE_SIZE];
    int3 *queries = new int3[N_QUERIES];

    generatePoints(points, N_POINTS);
    buildKDTree(points, tree, N_POINTS, TREE_SIZE);
    generatePoints(queries, N_QUERIES);


    auto start = std::chrono::system_clock::now();

    int3 *results = new int3[N_QUERIES];
    for (int i = 0; i < N_QUERIES; i++)
    {
        results[i] = findNearestNeighbor(tree, TREE_SIZE, 1, 0, queries[i]);
    }

    auto end = std::chrono::system_clock::now();
    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();

    printResults(queries, results, N_PRINT);
    std::cout << "Elapsed time in milliseconds : " << duration << "ms\n\n";
}

void print(int3 *points, int n)
{
    for (int i = 0; i < n; i++)
    {
        std::cout << "[" << points[i].x << ", " << points[i].y << ", " << points[i].z << "] ";
    }
    std::cout << std::endl;
}

void generatePoints(int3 *points, int n)
{
    for (int i = 0; i < n; i++)
    {
        points[i] = {.x = rand() % RANGE_MAX+1, .y = rand() % RANGE_MAX+1, .z = rand() % RANGE_MAX+1};
    }
}

int3 closer(int3 p, int3 p2, int3 p3)
{
    if ((abs(p.x - p2.x) + abs(p.y - p2.y) + abs(p.z - p2.z)) < (abs(p.x - p3.x) + abs(p.y - p3.y) + abs(p.z - p3.z)))
    {
        return p2;
    }
    return p3;
}

void buildSubTree(int3 *points, int3 *tree, int start, int end, int depth, int node)
{
    if (start >= end)
    {
        return;
    }

    std::sort(points + start, points + end, [depth](int3 p1, int3 p2) -> bool {
        if (depth % 3 == 0)
            return p1.x < p2.x;
        if (depth % 3 == 1)
            return p1.y < p2.y;
        return p1.z < p2.z;
    });

    int split = (start + end - 1) / 2;

    tree[node] = points[split];

    buildSubTree(points, tree, start, split, depth + 1, node * 2);
    buildSubTree(points, tree, split + 1, end, depth + 1, node * 2 + 1);
}

void buildKDTree(int3 *points, int3 *tree, int n, int treeSize)
{
    for (int i = 0; i < treeSize; i++)
    {
        tree[i] = {.x = INF, .y = -INF, .z = -INF};
    }

    buildSubTree(points, tree, 0, n, 0, 1);
}

int3 findNearestNeighbor(int3 *tree, int treeSize, int treeNode, int depth, int3 query)
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
            return closer(query, node, findNearestNeighbor(tree, treeSize, treeNode * 2, depth + 1, query));
        }
    }
    else if ((val1 > val2) && (treeNode * 2 + 1 < treeSize))
    {
        int3 rightChild = tree[treeNode * 2 + 1];
        if (rightChild.x != -INF && rightChild.y != -INF && rightChild.z != -INF)
        {
            return closer(query, node, findNearestNeighbor(tree, treeSize, treeNode * 2 + 1, depth + 1, query));
        }
    }
    return node;
}

void printResults(int3 *queries, int3 *results, int n) {
    for(int i = 0; i < n; i++) {
        std::cout<<"query: ["<<queries[i].x<<", "<<queries[i].y<<", "<<queries[i].z<<"] ";
        std::cout<<", result: ["<<results[i].x<<", "<<results[i].y<<", "<<results[i].z<<"] ";
        std::cout<<", distance: "<<sqrt(pow(queries[i].x - results[i].x, 2) + pow(queries[i].y - results[i].y, 2) + pow(queries[i].z - results[i].z, 2));
        std::cout<<std::endl;
    }
}