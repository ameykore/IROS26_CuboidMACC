# updated_search_algo.py
import heapq
from itertools import count
import timeit

from utility.exceptions import TimeoutError
from cpp import reachability  # NOTE: use the same reachability backend as problem_node

TIMEOUT = "TIMEOUT"
HIGH_LEVEL_FOUND = "FOUND"
NO_SOLUTION = "INFEASIBLE"


def a_star_search(start, goal, time_limit=float("inf"), w: float = 1.0, verbose: bool = False):
    """
    High-level A*/Weighted-A* over Assembly_Node states.

    Changes vs original:
      - Adds a real CLOSED-set policy via closed_best_g (best known g for popped/expanded states).
      - Uses g_best for OPEN to prevent pushing dominated duplicates.
      - Supports Weighted A* via f = g + w*h (w>=1.0).
    """

    if w < 1.0:
        raise ValueError(f"Weighted A* requires w >= 1.0, got w={w}")

    start_time = timeit.default_timer()
    tiebreaker = count()

    # Ensure goal occupancy grid is available (mirrors how your nodes use reachability.occupancy_grid)
    if goal.reachability.occupancy_grid is None:
        goal.reachability.occupancy_grid = reachability.occupancy_grid(
            goal.state, goal.x_dim, goal.y_dim, goal.z_dim
        )

    # Precompute scaffolding options once (your node caches this internally)
    start.get_scaffolding_options(goal)

    # --- Priority queue entry layout ---
    # (f, -g, -num_scaff_present, tie, node, parent_node)
    #
    # parent_node is stored for path reconstruction.
    #
    # We do NOT store an "obsolete flag"; instead we use g_best/closed_best_g to skip stale entries.
    open_heap = []

    # Best known g discovered so far for a node (OPEN candidates)
    g_best = {}

    # Best known g for nodes already expanded (CLOSED policy)
    closed_best_g = {}

    # Parent pointers for reconstruction: parent[node] = parent_node
    parent = {}

    # Stats (helpful to make the bottleneck undeniable)
    generated = 0
    expanded = 0
    stale_skips = 0
    reopened = 0

    def heuristic(n):
        return float(n.get_heuristic(goal))

    def g_cost(n):
        # Your node cost is path length (number of actions)
        return float(n.get_cost())

    def f_cost(g, h):
        return g + w * h

    # Push start
    g0 = g_cost(start)
    h0 = heuristic(start)
    entry0 = (f_cost(g0, h0), -g0, -len(start.scaffolding_present(goal)), next(tiebreaker), start, None)
    heapq.heappush(open_heap, entry0)
    g_best[start] = g0
    parent[start] = None

    while open_heap:
        # Timeout check
        if (timeit.default_timer() - start_time) >= time_limit:
            if verbose:
                print(f"[HL TIMEOUT] expanded={expanded} generated={generated} reopened={reopened} stale_skips={stale_skips}")
            raise TimeoutError("High Level Search timed out")

        f, ng, nsc, _, node, node_parent = heapq.heappop(open_heap)
        g = -ng  # stored as -g for tie-breaking

        # Stale entry check: if this is not the current best g for this node, skip it
        if node not in g_best or g != g_best[node]:
            stale_skips += 1
            continue

        # CLOSED-set dominance: if we have already expanded this node with <= g, skip
        prev_closed_g = closed_best_g.get(node, None)
        if prev_closed_g is not None and prev_closed_g <= g:
            stale_skips += 1
            continue

        # Mark expanded
        closed_best_g[node] = g
        expanded += 1

        # Goal test
        if node.is_goal_node(goal):
            if verbose:
                print("[HL FOUND] time:", timeit.default_timer() - start_time)
                print(f"expanded={expanded} generated={generated} reopened={reopened} stale_skips={stale_skips}")

            # Reconstruct plan as a list of nodes (matching your original output)
            plan_nodes = []
            cur = node
            while cur is not None and cur is not start:
                plan_nodes.append(cur)
                cur = parent.get(cur, None)
            plan_nodes.reverse()
            return HIGH_LEVEL_FOUND, plan_nodes

        # Expand children
        children = node.get_child_nodes(goal)

        for child in children:
            generated += 1
            cg = g_cost(child)

            # If we've already expanded child with a better g, don't bother
            prev_closed = closed_best_g.get(child, None)
            if prev_closed is not None and prev_closed <= cg:
                continue

            prev_open = g_best.get(child, None)
            if prev_open is None or cg < prev_open:
                if prev_open is not None:
                    reopened += 1

                g_best[child] = cg
                parent[child] = node

                ch = heuristic(child)
                cf = f_cost(cg, ch)
                entry = (cf, -cg, -len(child.scaffolding_present(goal)), next(tiebreaker), child, node)
                heapq.heappush(open_heap, entry)

    if verbose:
        print(f"[HL NO SOLUTION] expanded={expanded} generated={generated} reopened={reopened} stale_skips={stale_skips}")
    return NO_SOLUTION, []
