#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <random>

struct Edge
{
  int u, v;
};

struct CSRGraph
{
  int num_vertices;
  int num_edges;
  std::vector<int> row_ptr;
  std::vector<int> col_indices;
};

std::vector<Edge> generate_random_spanning_tree(int num_vertices, std::default_random_engine &generator);
void add_random_edges(std::vector<Edge> &edges, int num_vertices, int total_edges, std::default_random_engine &generator);
CSRGraph convert_to_csr(const std::vector<Edge> &edges, int num_vertices);
int verify_bfs(const CSRGraph &graph, const std::vector<int> &dist, const int source_vertex);

#endif // GRAPH_H