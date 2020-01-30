#include <iostream>
#include <chrono>
#include <time.h>
#include <algorithm>
#include <math.h>

const int N_POINTS = 1e3, N_QUERIES = 1e6, INF = 1e9;

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

int main()
{
    srand(16);

    int TREE_SIZE = 1;
    while (TREE_SIZE < N_POINTS)
        TREE_SIZE <<= 1;

    auto start = std::chrono::system_clock::now();

    int3 *points = new int3[N_POINTS];
    int3 *tree = new int3[TREE_SIZE];
    int3 *queries = new int3[N_QUERIES];

    generatePoints(points, N_POINTS);
    buildKDTree(points, tree, N_POINTS, TREE_SIZE);
    generatePoints(queries, N_QUERIES);


    int3 *results = new int3[N_QUERIES];
    for (int i = 0; i < N_QUERIES; i++)
    {
        results[i] = findNearestNeighbor(tree, TREE_SIZE, 1, 0, queries[i]);
    }

    auto end = std::chrono::system_clock::now();
    float duration = 1000.0 * std::chrono::duration<float>(end - start).count();

    // std::cout<<"Queries: \n";
    // print(queries, N_QUERIES);
    // std::cout<<"Nearest neighbors: \n";
    // print(results, N_QUERIES);
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
        points[i] = {.x = rand() % 100, .y = rand() % 100, .z = rand() % 100};
    }
}

int3 closer(int3 p, int3 p2, int3 p3)
{
    if ((pow(p.x - p2.x, 2) + pow(p.y - p2.y, 2) + pow(p.z - p2.z, 2)) < (pow(p.x - p3.x, 2) + pow(p.y - p3.y, 2) + pow(p.z - p3.z, 2)))
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
        int3 leftChild = tree[treeSize * 2];
        if (leftChild.x != -INF && leftChild.y != -INF && leftChild.z != -INF)
        {
            return closer(query, node, findNearestNeighbor(tree, treeSize, treeNode * 2, depth + 1, query));
        }
    }
    else if ((val1 > val2) && (treeNode * 2 + 1 < treeSize))
    {
        int3 rightChild = tree[treeSize * 2];
        if (rightChild.x != -INF && rightChild.y != -INF && rightChild.z != -INF)
        {
            return closer(query, node, findNearestNeighbor(tree, treeSize, treeNode * 2 + 1, depth + 1, query));
        }
    }
    return node;
}