#include "bfs.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstddef>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list) { list->count = 0; }

void vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                   int *distances, int *mf, int *new_dist) {
  int outgoing_sum = 0;

#pragma omp parallel for reduction(+ : outgoing_sum)
  for (int i = 0; i < frontier->count; i++) {
    int node = frontier->vertices[i];

    int start_edge = g->outgoing_starts[node];
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                              : g->outgoing_starts[node + 1];

    // attempt to add all neighbors to the new frontier
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int outgoing = g->outgoing_edges[neighbor];

      if (distances[outgoing] == NOT_VISITED_MARKER) {
        if (__sync_bool_compare_and_swap(&distances[outgoing],
                                         NOT_VISITED_MARKER,
                                         distances[node] + 1)) {
          int index = __sync_fetch_and_add(&new_frontier->count, 1);
          new_frontier->vertices[index] = outgoing;
          outgoing_sum += outgoing_size(g, outgoing);
          new_dist[outgoing] = distances[node] + 1;
        }
      }
    }
  }

  *mf = outgoing_sum;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  int *new_distances = (int *)malloc(sizeof(int) * graph->num_nodes);

  // initialize all nodes to NOT_VISITED
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
    new_distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  int mf = 0;

  while (frontier->count != 0) {
#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif
    vertex_set_clear(new_frontier);
    top_down_step(graph, frontier, new_frontier, sol->distances, &mf,
                  new_distances);
#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

int bottom_up_step(Graph g, int *distances, int *mf, int dist, int *old_dist,
                   int *new_dist) {
  int outgoing_sum = 0;
  int nf = 0;

#pragma omp parallel for reduction(+ : outgoing_sum, nf)
  for (int i = 0; i < g->num_nodes; i++) {
    if (distances[i] == NOT_VISITED_MARKER) {
      int start_edge = g->incoming_starts[i];
      int end_edge =
          (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];

      for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
        int incoming = g->incoming_edges[neighbor];

        if (old_dist[incoming] != NOT_VISITED_MARKER) {
          new_dist[i] = dist + 1;
          distances[i] = dist + 1;
          outgoing_sum += outgoing_size(g, i);
          nf++;
          break;
        }
      }
    }
  }

  *mf = outgoing_sum;
  return nf;
}

void bfs_bottom_up(Graph graph, solution *sol) {
  int *distances = (int *)malloc(sizeof(int) * graph->num_nodes);
  int *new_distances = (int *)malloc(sizeof(int) * graph->num_nodes);

  // initialize all nodes to NOT_VISITED
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
    distances[i] = NOT_VISITED_MARKER;
    new_distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  sol->distances[ROOT_NODE_ID] = 0;
  distances[ROOT_NODE_ID] = 0;
  new_distances[ROOT_NODE_ID] = 0;

  int dist = 0, mf = 1;

  while (mf != 0) {
    bottom_up_step(graph, sol->distances, &mf, dist++, distances,
                   new_distances);

    int *tmp = distances;
    distances = new_distances;
    new_distances = tmp;
  }
}

void bfs_hybrid(Graph graph, solution *sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  int *distances = (int *)malloc(sizeof(int) * graph->num_nodes);
  int *new_distances = (int *)malloc(sizeof(int) * graph->num_nodes);

  // initialize all nodes to NOT_VISITED
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
    distances[i] = NOT_VISITED_MARKER;
    new_distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
  distances[ROOT_NODE_ID] = 0;
  new_distances[ROOT_NODE_ID] = 0;

  constexpr int ALPHA = 14, BETA = 24;

  bool use_top_down = true, is_growing = true;
  int mf = 0, nf = frontier->count, mu = graph->num_nodes;
  int dist = 0, pre_nf;

  while (frontier->count != 0) {
    pre_nf = nf;

    if (use_top_down && mf > mu / ALPHA && is_growing) {
      use_top_down = false;
    } else if (!use_top_down && nf < graph->num_nodes / BETA && !is_growing) {
      vertex_set_clear(frontier);

      for (int i = 0; i < graph->num_nodes; i++) {
        if (sol->distances[i] == dist) {
          frontier->vertices[frontier->count++] = i;
        }
      }
      use_top_down = true;
    }

    nf = frontier->count;
    mu -= mf;
    mf = 0;

    if (use_top_down) {
      vertex_set_clear(new_frontier);
      top_down_step(graph, frontier, new_frontier, sol->distances, &mf,
                    new_distances);

      vertex_set *tmp = frontier;
      frontier = new_frontier;
      new_frontier = tmp;
    } else {
      nf = bottom_up_step(graph, sol->distances, &mf, dist, distances,
                          new_distances);
      frontier->count = nf;
    }

    int *tmp = distances;
    distances = new_distances;
    new_distances = tmp;

    dist++;

    is_growing = (nf > pre_nf);
  }
}