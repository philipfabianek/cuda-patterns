#include "graph.h"
#include <algorithm>
#include <queue>
#include <set>

std::vector<Edge> generate_random_spanning_tree(int num_vertices, std::default_random_engine &generator)
{
  std::vector<Edge> edges;
  edges.reserve(num_vertices - 1);

  // Allocate a vector of visited vertices, include the starting node 0
  std::vector<int> visited;
  visited.reserve(num_vertices);
  visited.push_back(0);

  // Allocate a vector of unvisited vertices filled with 1, ..., num_vertices-1
  std::vector<int> unvisited(num_vertices - 1);
  std::iota(unvisited.begin(), unvisited.end(), 1);

  // Connect unvisited vertices to the visited set randomly
  std::uniform_int_distribution<int> unvisited_dist(0, unvisited.size() - 1);
  while (!unvisited.empty())
  {
    // Pick a random vertex from the visited set
    std::uniform_int_distribution<int> visited_dist(0, visited.size() - 1);
    int u = visited[visited_dist(generator)];

    // Pick a random vertex from the unvisited set
    unvisited_dist.param(std::uniform_int_distribution<int>::param_type(0, unvisited.size() - 1));
    int v_idx = unvisited_dist(generator);
    int v = unvisited[v_idx];

    // Add an edge between them
    edges.push_back({u, v});

    // Move the vertex from unvisited to visited
    visited.push_back(v);
    std::swap(unvisited[v_idx], unvisited.back());
    unvisited.pop_back();
  }

  return edges;
}

void add_random_edges(std::vector<Edge> &edges, int num_vertices, int total_edges, std::default_random_engine &generator)
{
  // Use this to efficiently check whether an edge already exists
  std::set<std::pair<int, int>> existing_edges;
  for (const auto &edge : edges)
  {
    existing_edges.insert({std::min(edge.u, edge.v), std::max(edge.u, edge.v)});
  }

  std::uniform_int_distribution<int> vertex_dist(0, num_vertices - 1);

  // Until there are <total_edges> total edges randomly pick
  // two vertices and insert an edge between them
  while (edges.size() < total_edges)
  {
    int u = vertex_dist(generator);
    int v = vertex_dist(generator);

    if (u != v)
    {
      std::pair<int, int> new_edge = {std::min(u, v), std::max(u, v)};
      if (existing_edges.find(new_edge) == existing_edges.end())
      {
        edges.push_back({u, v});
        existing_edges.insert(new_edge);
      }
    }
  }
}

CSRGraph convert_to_csr(const std::vector<Edge> &edges, int num_vertices)
{
  CSRGraph graph;
  graph.num_vertices = num_vertices;
  graph.num_edges = edges.size();

  // First build an adjacency list from the edge list
  std::vector<std::vector<int>> adj_list(num_vertices);
  for (const auto &edge : edges)
  {
    adj_list[edge.u].push_back(edge.v);
    adj_list[edge.v].push_back(edge.u);
  }

  graph.row_ptr.resize(num_vertices + 1);
  graph.row_ptr[0] = 0;

  // There will be 2 * col_indices entries for undirected edges
  graph.col_indices.reserve(edges.size() * 2);

  // For each vertex add the relevant column indices for each neighbor
  for (int i = 0; i < num_vertices; ++i)
  {
    // This can help with memory access?
    // std::sort(adj_list[i].begin(), adj_list[i].end());

    for (int neighbor : adj_list[i])
    {
      graph.col_indices.push_back(neighbor);
    }

    graph.row_ptr[i + 1] = graph.col_indices.size();
  }

  return graph;
}

/*
 * Verifies the result of the BFS computation from the GPU
 * by comparing it against a traditional CPU-based computation.
 */
int verify_bfs(const CSRGraph &graph, const std::vector<int> &dist, const int source_vertex)
{
  int status = 0;

  std::vector<int> cpu_dist(graph.num_vertices, -1);
  cpu_dist[source_vertex] = 0;
  std::queue<int> q;
  q.push(source_vertex);

  while (!q.empty())
  {
    int u = q.front();
    q.pop();

    for (int i = graph.row_ptr[u]; i < graph.row_ptr[u + 1]; ++i)
    {
      int v = graph.col_indices[i];
      if (cpu_dist[v] == -1)
      {
        cpu_dist[v] = cpu_dist[u] + 1;
        q.push(v);
      }
    }
  }

  for (int i = 0; i < graph.num_vertices; ++i)
  {
    if (cpu_dist[i] != dist[i])
    {
      fprintf(stderr, "Mismatch at vertex %d: expected %d, got %d\n", i, cpu_dist[i], dist[i]);
      status = -1;
    }
  }

  return status;
}