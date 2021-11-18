#include "page_rank.h"

#include <omp.h>
#include <stdlib.h>

#include <cmath>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is
// num_nodes(g)) damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence) {
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  /*
      For PP students: Implement the page rank algorithm here.  You
      are expected to parallelize the algorithm using openMP.  Your
      solution may need to allocate (and free) temporary arrays.

      Basic page rank pseudocode is provided below to get you started:

      // initialization: see example code above
      score_old[vi] = 1/numNodes;

      while (!converged) {

        // compute score_new[vi] for all nodes vi:
        score_new[vi] = sum over all nodes vj reachable from incoming edges
                           { score_old[vj] / number of edges leaving vj  }
        score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

        score_new[vi] += sum over all nodes v in graph with no outgoing edges
                           { damping * score_old[v] / numNodes }

        // compute how much per-node scores have changed
        // quit once algorithm has converged

        global_diff = sum over all nodes vi { abs(score_new[vi] -
        score_old[vi])
      }; converged = (global_diff < convergence)
      }

  */

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

#pragma omp parallel for
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  double* score_new = new double[numNodes];
  double global_diff, sum_no_outgoing;

  do {
    global_diff = 0.0;
    sum_no_outgoing = 0.0;
    
#pragma omp parallel
    {
#pragma omp for reduction(+ : sum_no_outgoing)
      for (int j = 0; j < numNodes; ++j) {
        sum_no_outgoing += outgoing_size(g, j) == 0 ? solution[j] : 0;
      }

#pragma omp for
      for (int i = 0; i < numNodes; ++i) {
        const Vertex* start = incoming_begin(g, i);
        const Vertex* end = incoming_end(g, i);

        double sum_reachable = 0.0;
        for (const Vertex* v = start; v != end; v++) {
          sum_reachable += solution[*v] / outgoing_size(g, *v);
        }

        score_new[i] = (damping * sum_reachable) + (1.0 - damping) / numNodes +
                       damping * sum_no_outgoing / numNodes;
      }

#pragma omp for reduction(+ : global_diff)
      for (int i = 0; i < numNodes; ++i) {
        global_diff += fabs(score_new[i] - solution[i]);
        solution[i] = score_new[i];
      }
    }
  } while (global_diff >= convergence);

  delete[] score_new;
}
