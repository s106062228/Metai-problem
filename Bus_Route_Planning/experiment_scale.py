"""
Multi-Scale Experiment Runner — 50 to 10M Passengers
=====================================================
Self-contained scalability study. Does NOT import bus_route_optimizer.py
functions that hardcode grid_size=100. All core routines are generalized.

Generates 4 figures:
  fig_scale_time.png       — Log-log: passengers vs. computation time
  fig_scale_cost_per_pax.png — Log: passengers vs. cost / passenger
  fig_scale_routes.png     — 2x2 route maps at 4 scales
  fig_scale_stops.png      — Log: passengers vs. optimal stop count
"""

import os
import sys
import time
import heapq
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Cost parameters (same as bus_route_optimizer.py) ──
ALPHA = 1.5
BETA = 3.0
DELTA = 2.0
C_FIXED = 10.0
C_TERRAIN = 3.0

# ── Passenger sample cap for optimization iterations ──
PAX_SAMPLE_CAP = 5000


# =============================================================================
# Scalable Environment
# =============================================================================

class ScalableEnvironment:
    """Environment that scales proportionally from the 100×100 base case."""

    # Base terrain blob parameters (for grid_size=100)
    BASE_BLOBS = [
        (20, 70, 12, 2.0),
        (60, 20, 10, 1.5),
        (80, 80, 8, 2.5),
    ]
    # Base cluster parameters (cx, cy, std, fraction_of_total)
    BASE_CLUSTERS = [
        (15, 30, 8, 0.24),
        (50, 45, 10, 0.26),
        (75, 65, 7, 0.20),
        (40, 80, 9, 0.16),
        (85, 25, 6, 0.14),
    ]

    def __init__(self, grid_size=100, n_passengers=50, seed=42):
        np.random.seed(seed)
        self.grid_size = grid_size
        self.n_passengers = n_passengers
        self.scale = grid_size / 100.0

        # A and B scale proportionally
        self.A = np.array([5.0, 10.0]) * self.scale
        self.B = np.array([92.0, 88.0]) * self.scale

        # Restricted zones — scale proportionally
        self.restricted_zones = [
            {'type': 'rect',
             'x': 30 * self.scale, 'y': 20 * self.scale,
             'w': 15 * self.scale, 'h': 20 * self.scale},
            {'type': 'rect',
             'x': 55 * self.scale, 'y': 55 * self.scale,
             'w': 12 * self.scale, 'h': 18 * self.scale},
            {'type': 'circle',
             'cx': 70 * self.scale, 'cy': 35 * self.scale,
             'r': 10 * self.scale},
        ]

        # Passengers — clustered, scaled
        self.passengers = self._generate_passengers()

        # Terrain blobs scaled (no stored grid — computed analytically)
        self.terrain_blobs = [
            (cx * self.scale, cy * self.scale, sigma * self.scale, intensity)
            for cx, cy, sigma, intensity in self.BASE_BLOBS
        ]

    def _generate_passengers(self):
        passengers = []
        for cx, cy, std, frac in self.BASE_CLUSTERS:
            n = max(1, int(self.n_passengers * frac))
            pts = np.column_stack([
                np.random.normal(cx * self.scale, std * self.scale, n),
                np.random.normal(cy * self.scale, std * self.scale, n),
            ])
            passengers.append(pts)
        passengers = np.vstack(passengers)
        passengers = np.clip(passengers, 1, self.grid_size - 1)
        # Trim or pad to exact count
        if len(passengers) >= self.n_passengers:
            passengers = passengers[:self.n_passengers]
        else:
            # Duplicate random passengers to reach target
            extra = self.n_passengers - len(passengers)
            idx = np.random.choice(len(passengers), extra, replace=True)
            jitter = np.random.normal(0, 0.5 * self.scale, (extra, 2))
            passengers = np.vstack([passengers,
                                    np.clip(passengers[idx] + jitter,
                                            1, self.grid_size - 1)])
        return passengers

    def is_in_restricted(self, x, y):
        for zone in self.restricted_zones:
            if zone['type'] == 'rect':
                if (zone['x'] <= x <= zone['x'] + zone['w'] and
                        zone['y'] <= y <= zone['y'] + zone['h']):
                    return True
            elif zone['type'] == 'circle':
                if ((x - zone['cx'])**2 + (y - zone['cy'])**2
                        <= zone['r']**2):
                    return True
        return False

    def get_terrain_cost_at(self, x, y):
        """Analytical terrain cost — no stored grid needed."""
        val = 1.0
        for cx, cy, sigma, intensity in self.terrain_blobs:
            val += intensity * np.exp(
                -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return val

    def get_terrain_grid_lowres(self, res=200):
        """Low-res terrain grid for visualization only."""
        size = min(res, self.grid_size)
        step = self.grid_size / size
        yy, xx = np.mgrid[0:size, 0:size]
        grid = np.ones((size, size))
        for cx, cy, sigma, intensity in self.terrain_blobs:
            blob = intensity * np.exp(
                -(((xx * step) - cx)**2 + ((yy * step) - cy)**2)
                / (2 * sigma**2))
            grid += blob
        return grid, step


# =============================================================================
# Generalized A*
# =============================================================================

def snap_to_valid(env, point, search_radius=15):
    """Snap a point to the nearest non-restricted location."""
    x, y = point
    if not env.is_in_restricted(x, y):
        return np.array([x, y])
    best, best_dist = None, float('inf')
    sr = int(search_radius)
    for dx in range(-sr, sr + 1):
        for dy in range(-sr, sr + 1):
            nx, ny = x + dx, y + dy
            if (0 <= nx < env.grid_size and 0 <= ny < env.grid_size
                    and not env.is_in_restricted(nx, ny)):
                d = dx**2 + dy**2
                if d < best_dist:
                    best_dist = d
                    best = np.array([float(nx), float(ny)])
    return best


def astar(env, start, goal, grid_res=1):
    """A* on discretized grid, generalized for any grid_size."""
    res = grid_res
    sx = int(round(start[0] / res))
    sy = int(round(start[1] / res))
    gx = int(round(goal[0] / res))
    gy = int(round(goal[1] / res))
    max_c = int(env.grid_size / res)

    def h(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) * res

    heap = [(h((sx, sy), (gx, gy)), 0.0, sx, sy)]
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[(sx, sy)] = 0.0
    closed = set()

    while heap:
        f, g, x, y = heapq.heappop(heap)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        if x == gx and y == gy:
            path = []
            node = (gx, gy)
            while node in came_from:
                path.append(np.array([node[0] * res, node[1] * res],
                                     dtype=float))
                node = came_from[node]
            path.append(np.array([node[0] * res, node[1] * res],
                                 dtype=float))
            path.reverse()
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx <= max_c and 0 <= ny <= max_c:
                rx, ry = nx * res, ny * res
                if env.is_in_restricted(rx, ry):
                    continue
                mc = np.sqrt(dx**2 + dy**2) * res
                ng = g + mc
                if ng < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = ng
                    came_from[(nx, ny)] = (x, y)
                    heapq.heappush(heap,
                                   (ng + h((nx, ny), (gx, gy)), ng, nx, ny))
    return None


def smooth_route(route, env, n_samples=30):
    """Line-of-sight route smoothing."""
    if len(route) <= 2:
        return route
    smoothed = [route[0]]
    i = 0
    while i < len(route) - 1:
        best_j = i + 1
        for j in range(len(route) - 1, i, -1):
            if not _line_intersects(route[i], route[j], env, n_samples):
                best_j = j
                break
        smoothed.append(route[best_j])
        i = best_j
    return np.array(smoothed)


def _line_intersects(p1, p2, env, n_samples=25):
    for t in np.linspace(0, 1, n_samples):
        pt = p1 + t * (p2 - p1)
        if env.is_in_restricted(pt[0], pt[1]):
            return True
    return False


def order_stops_along_AB(stops, env):
    ab = env.B - env.A
    ab_n = ab / (np.linalg.norm(ab) + 1e-9)
    idx_proj = [(np.dot(s - env.A, ab_n), i) for i, s in enumerate(stops)]
    idx_proj.sort()
    return np.array([stops[i] for _, i in idx_proj])


def build_route(env, stops, grid_res=1):
    waypoints = [env.A]
    if len(stops) > 0:
        ordered = order_stops_along_AB(stops, env)
        waypoints.extend(list(ordered))
    waypoints.append(env.B)
    full = []
    for i in range(len(waypoints) - 1):
        seg = astar(env, waypoints[i], waypoints[i + 1], grid_res)
        if seg is None:
            seg = [waypoints[i], waypoints[i + 1]]
        if i > 0 and len(full) > 0:
            seg = seg[1:]
        full.extend(seg)
    return np.array(full)


# =============================================================================
# Fast Walking Cost — Vectorized
# =============================================================================

def walking_cost_vectorized(passengers, all_stops):
    """O(n) memory per stop. Returns total walking distance."""
    min_dists = np.full(len(passengers), np.inf)
    for s in all_stops:
        d = np.sqrt(((passengers - s)**2).sum(axis=1))
        np.minimum(min_dists, d, out=min_dists)
    return float(min_dists.sum())


def compute_costs_fast(env, stops, route, passengers):
    """Fast cost computation with vectorized walking cost."""
    # Route length
    if len(route) > 1:
        diffs = route[1:] - route[:-1]
        route_len = float(np.sqrt((diffs**2).sum(axis=1)).sum())
    else:
        route_len = 0.0
    route_cost = ALPHA * route_len

    # Station cost
    station_cost = 0.0
    for s in stops:
        gamma = env.get_terrain_cost_at(s[0], s[1])
        station_cost += C_FIXED + C_TERRAIN * gamma
    station_cost *= BETA

    # Walking cost
    all_stops = np.vstack([env.A] + list(stops) + [env.B]) if len(stops) > 0 \
        else np.vstack([env.A, env.B])
    walk_total = walking_cost_vectorized(passengers, all_stops)
    walking_cost = DELTA * walk_total

    total = route_cost + station_cost + walking_cost
    n_pax = len(passengers)
    return {
        'total': total,
        'route_cost': route_cost,
        'station_cost': station_cost,
        'walking_cost': walking_cost,
        'route_length': route_len,
        'n_stops': len(stops),
        'avg_walk': walk_total / max(n_pax, 1),
        'max_walk': 0.0,  # skip for speed at scale
    }


# =============================================================================
# Candidate Stop Generation
# =============================================================================

def generate_candidates(env, target_n=12, pax_for_cluster=None):
    """Generate candidate stops via KMeans on passengers."""
    if pax_for_cluster is None:
        pax_for_cluster = env.passengers
    # Subsample for KMeans if too large
    if len(pax_for_cluster) > 10000:
        from sklearn.cluster import MiniBatchKMeans as KM
        idx = np.random.choice(len(pax_for_cluster), 10000, replace=False)
        pax_sub = pax_for_cluster[idx]
    else:
        from sklearn.cluster import KMeans as KM
        pax_sub = pax_for_cluster

    all_centers = []
    for k in [3, 5, 7, 9]:
        if k > len(pax_sub):
            continue
        km = KM(n_clusters=k, random_state=42, n_init=5).fit(pax_sub)
        all_centers.extend(km.cluster_centers_)

    # Density-based candidates: far from A and B
    dists_to_AB = np.minimum(
        np.linalg.norm(pax_sub - env.A, axis=1),
        np.linalg.norm(pax_sub - env.B, axis=1))
    far_idx = np.argsort(dists_to_AB)[-6:]
    all_centers.extend(pax_sub[far_idx])

    snapped = []
    min_sep = 5 * env.scale
    for c in all_centers:
        s = snap_to_valid(env, c, search_radius=15 * env.scale)
        if s is not None:
            if all(np.linalg.norm(s - ex) >= min_sep for ex in snapped):
                snapped.append(s)

    candidates = np.array(snapped[:target_n])
    return candidates


# =============================================================================
# Distance Cache
# =============================================================================

def build_distance_cache(env, candidates, grid_res=1):
    """Pre-compute A* distances between all waypoint pairs."""
    waypoints = [env.A] + list(candidates) + [env.B]
    n = len(waypoints)
    dist_cache = np.full((n, n), np.inf)
    path_cache = {}

    for i in range(n):
        dist_cache[i][i] = 0.0
        for j in range(i + 1, n):
            path = astar(env, waypoints[i], waypoints[j], grid_res)
            if path is not None:
                sm = smooth_route(np.array(path), env)
                d = float(np.sqrt(((sm[1:] - sm[:-1])**2).sum(axis=1)).sum())
            else:
                d = np.linalg.norm(waypoints[i] - waypoints[j])
                sm = np.array([waypoints[i], waypoints[j]])
            dist_cache[i][j] = d
            dist_cache[j][i] = d
            path_cache[(i, j)] = sm
            path_cache[(j, i)] = sm[::-1]

    return dist_cache, path_cache, waypoints


# =============================================================================
# Quick Cost Evaluation (cached distances)
# =============================================================================

def quick_cost(env, selected_indices, candidates, dist_cache, passengers,
               n_total=None):
    """Fast cost evaluation using cached distances. Supports sampling."""
    n_cand = len(candidates)
    stops = candidates[selected_indices] if len(selected_indices) > 0 \
        else np.empty((0, 2))
    scale_factor = 1.0
    if n_total is not None and n_total > len(passengers):
        scale_factor = n_total / len(passengers)

    # Route cost
    if len(stops) == 0:
        route_len = dist_cache[0][n_cand + 1]
    else:
        ab = env.B - env.A
        ab_n = ab / (np.linalg.norm(ab) + 1e-9)
        projs = [(np.dot(stops[i] - env.A, ab_n), selected_indices[i])
                 for i in range(len(stops))]
        projs.sort()
        wp = [0] + [p[1] + 1 for p in projs] + [n_cand + 1]
        route_len = sum(dist_cache[wp[i]][wp[i + 1]]
                        for i in range(len(wp) - 1))
    route_cost = ALPHA * route_len

    # Station cost
    station_cost = BETA * sum(
        C_FIXED + C_TERRAIN * env.get_terrain_cost_at(s[0], s[1])
        for s in stops)

    # Walking cost (vectorized)
    all_s = np.vstack([env.A] + list(stops) + [env.B]) if len(stops) > 0 \
        else np.vstack([env.A, env.B])
    walk_total = walking_cost_vectorized(passengers, all_s)
    walking_cost = DELTA * walk_total * scale_factor

    return route_cost + station_cost + walking_cost


# =============================================================================
# Full Evaluation
# =============================================================================

def full_evaluate(env, selected_indices, candidates, dist_cache, path_cache,
                  passengers, n_total=None):
    """Full evaluation with route reconstruction."""
    n_cand = len(candidates)
    stops = candidates[selected_indices] if len(selected_indices) > 0 \
        else np.empty((0, 2))
    scale_factor = 1.0
    if n_total is not None and n_total > len(passengers):
        scale_factor = n_total / len(passengers)

    if len(stops) == 0:
        route = path_cache.get((0, n_cand + 1),
                               np.array([env.A, env.B]))
    else:
        ab = env.B - env.A
        ab_n = ab / (np.linalg.norm(ab) + 1e-9)
        projs = [(np.dot(stops[i] - env.A, ab_n), selected_indices[i], i)
                 for i in range(len(stops))]
        projs.sort()
        cache_wp = [0] + [p[1] + 1 for p in projs] + [n_cand + 1]
        ordered_stop_idx = [p[2] for p in projs]
        stops = stops[ordered_stop_idx]
        segments = []
        for i in range(len(cache_wp) - 1):
            seg = path_cache.get((cache_wp[i], cache_wp[i + 1]))
            if seg is None:
                all_wp = [env.A] + list(candidates) + [env.B]
                seg = np.array([all_wp[cache_wp[i]], all_wp[cache_wp[i + 1]]])
            if i > 0 and len(segments) > 0:
                seg = seg[1:]
            segments.append(seg)
        route = np.vstack(segments) if segments else np.array([env.A, env.B])

    # Cost
    all_s = np.vstack([env.A] + list(stops) + [env.B]) if len(stops) > 0 \
        else np.vstack([env.A, env.B])

    if len(route) > 1:
        diffs = route[1:] - route[:-1]
        route_len = float(np.sqrt((diffs**2).sum(axis=1)).sum())
    else:
        route_len = 0.0

    station_cost = BETA * sum(
        C_FIXED + C_TERRAIN * env.get_terrain_cost_at(s[0], s[1])
        for s in stops)

    walk_total = walking_cost_vectorized(passengers, all_s)
    walking_cost = DELTA * walk_total * scale_factor

    total = ALPHA * route_len + station_cost + walking_cost
    n_pax = len(passengers)

    return {
        'stops': stops,
        'route': route,
        'total': total,
        'route_cost': ALPHA * route_len,
        'station_cost': station_cost,
        'walking_cost': walking_cost,
        'route_length': route_len,
        'n_stops': len(stops),
        'avg_walk': walk_total / max(n_pax, 1),
    }


# =============================================================================
# Algorithm Implementations
# =============================================================================

def run_exact(env, candidates, dist_cache, path_cache, passengers,
              n_total, max_stops=8):
    """Exact enumeration over all subsets."""
    n = len(candidates)
    best_cost = float('inf')
    best_indices = []
    t0 = time.time()
    for k in range(1, min(max_stops, n) + 1):
        for combo in combinations(range(n), k):
            indices = list(combo)
            cost = quick_cost(env, indices, candidates, dist_cache,
                              passengers, n_total)
            if cost < best_cost:
                best_cost = cost
                best_indices = indices
    elapsed = time.time() - t0
    result = full_evaluate(env, best_indices, candidates, dist_cache,
                           path_cache, passengers, n_total)
    return result, elapsed


def run_two_stage(env, passengers, n_total, grid_res=1):
    """Simplified two-stage heuristic adapted for scale."""
    t0 = time.time()

    # Stage 1: KMeans initial placement
    pax_sub = passengers
    if len(passengers) > 10000:
        from sklearn.cluster import MiniBatchKMeans as KM
        idx = np.random.choice(len(passengers), 10000, replace=False)
        pax_sub = passengers[idx]
    else:
        from sklearn.cluster import KMeans as KM

    n_clusters = min(10, len(pax_sub))
    km = KM(n_clusters=n_clusters, random_state=42, n_init=5).fit(pax_sub)
    stops = []
    for c in km.cluster_centers_:
        s = snap_to_valid(env, c, search_radius=15 * env.scale)
        if s is not None:
            stops.append(s)
    if len(stops) == 0:
        stops = [env.A + (env.B - env.A) * 0.5]
    stops = np.array(stops)

    # Build route + evaluate
    route = build_route(env, stops, grid_res)
    route = smooth_route(route, env)
    costs = compute_costs_fast(env, stops, route, passengers)

    # Scale walking cost
    if n_total > len(passengers):
        sf = n_total / len(passengers)
        costs['walking_cost'] *= sf
        costs['total'] = (costs['route_cost'] + costs['station_cost']
                          + costs['walking_cost'])

    # Stage 2: Simple refinement — prune stops serving few passengers
    all_s = np.vstack([env.A] + list(stops) + [env.B])
    dists_to_stops = np.array([
        np.linalg.norm(passengers - s, axis=1) for s in all_s]).T
    assignments = dists_to_stops.argmin(axis=1)
    counts = defaultdict(int)
    for a in assignments:
        counts[a] += 1

    min_pax = max(2, int(0.05 * len(passengers)))
    keep = []
    for i, s in enumerate(stops):
        if counts[i + 1] >= min_pax:
            keep.append(s)
    if len(keep) == 0:
        keep = [stops[0]]
    stops = np.array(keep)

    # Centroid relocation
    all_s = np.vstack([env.A] + list(stops) + [env.B])
    dists_to_stops = np.array([
        np.linalg.norm(passengers - s, axis=1) for s in all_s]).T
    assignments = dists_to_stops.argmin(axis=1)
    for i in range(len(stops)):
        mask = assignments == (i + 1)
        if mask.sum() >= 2:
            centroid = passengers[mask].mean(axis=0)
            centroid = np.clip(centroid, 1, env.grid_size - 1)
            if not env.is_in_restricted(centroid[0], centroid[1]):
                stops[i] = centroid

    # Re-route and evaluate
    route = build_route(env, stops, grid_res)
    route = smooth_route(route, env)
    costs = compute_costs_fast(env, stops, route, passengers)
    if n_total > len(passengers):
        sf = n_total / len(passengers)
        costs['walking_cost'] *= sf
        costs['total'] = (costs['route_cost'] + costs['station_cost']
                          + costs['walking_cost'])

    elapsed = time.time() - t0
    return {
        'stops': stops,
        'route': route,
        **costs,
    }, elapsed


def run_ga(env, candidates, dist_cache, path_cache, passengers, n_total,
           pop_size=40, n_gen=80, p_cross=0.8, p_mut=0.15, seed=123):
    """Binary-encoded GA."""
    np.random.seed(seed)
    n = len(candidates)

    population = []
    for _ in range(pop_size):
        k = np.random.randint(1, min(8, n) + 1)
        chromo = np.zeros(n, dtype=int)
        chromo[np.random.choice(n, k, replace=False)] = 1
        population.append(chromo)

    def fitness(chromo):
        idx = list(np.where(chromo == 1)[0])
        if len(idx) == 0:
            idx = [np.random.randint(n)]
        return quick_cost(env, idx, candidates, dist_cache, passengers,
                          n_total)

    costs = [fitness(c) for c in population]
    best_cost = min(costs)
    best_chromo = population[int(np.argmin(costs))].copy()

    t0 = time.time()
    for gen in range(n_gen):
        new_pop = []
        ranked = np.argsort(costs)
        new_pop.append(population[ranked[0]].copy())
        new_pop.append(population[ranked[1]].copy())

        while len(new_pop) < pop_size:
            def tourney():
                ti = np.random.choice(pop_size, 3, replace=False)
                return population[ti[np.argmin(
                    [costs[i] for i in ti])]].copy()

            p1, p2 = tourney(), tourney()
            if np.random.random() < p_cross:
                mask = np.random.randint(0, 2, n)
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                for i in range(n):
                    if np.random.random() < p_mut:
                        child[i] = 1 - child[i]
                if child.sum() == 0:
                    child[np.random.randint(n)] = 1
                while child.sum() > 8:
                    ones = np.where(child == 1)[0]
                    child[np.random.choice(ones)] = 0

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop[:pop_size]
        costs = [fitness(c) for c in population]

        gen_best = min(costs)
        if gen_best < best_cost:
            best_cost = gen_best
            best_chromo = population[int(np.argmin(costs))].copy()

    elapsed = time.time() - t0
    best_idx = list(np.where(best_chromo == 1)[0])
    result = full_evaluate(env, best_idx, candidates, dist_cache, path_cache,
                           passengers, n_total)
    return result, elapsed


def run_sa(env, candidates, dist_cache, path_cache, passengers, n_total,
           T0=200, cooling=0.995, n_iter=1500, seed=456):
    """Simulated Annealing with binary vector."""
    np.random.seed(seed)
    n = len(candidates)

    current = np.zeros(n, dtype=int)
    k = np.random.randint(3, min(6, n) + 1)
    current[np.random.choice(n, k, replace=False)] = 1

    current_cost = quick_cost(env, list(np.where(current == 1)[0]),
                              candidates, dist_cache, passengers, n_total)
    best = current.copy()
    best_cost = current_cost
    T = T0

    t0 = time.time()
    for it in range(n_iter):
        neighbor = current.copy()
        move = np.random.random()
        if move < 0.35:
            neighbor[np.random.randint(n)] ^= 1
        elif move < 0.65:
            ones = np.where(neighbor == 1)[0]
            zeros = np.where(neighbor == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                neighbor[np.random.choice(ones)] = 0
                neighbor[np.random.choice(zeros)] = 1
        else:
            i1, i2 = np.random.choice(n, 2, replace=False)
            neighbor[i1] ^= 1
            neighbor[i2] ^= 1

        if neighbor.sum() == 0:
            neighbor[np.random.randint(n)] = 1
        while neighbor.sum() > 8:
            ones = np.where(neighbor == 1)[0]
            neighbor[np.random.choice(ones)] = 0

        idx = list(np.where(neighbor == 1)[0])
        neighbor_cost = quick_cost(env, idx, candidates, dist_cache,
                                   passengers, n_total)
        delta = neighbor_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(
                -delta / max(T, 1e-10)):
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost
        T *= cooling

    elapsed = time.time() - t0
    best_idx = list(np.where(best == 1)[0])
    result = full_evaluate(env, best_idx, candidates, dist_cache, path_cache,
                           passengers, n_total)
    return result, elapsed


def run_aco(env, candidates, dist_cache, path_cache, passengers, n_total,
            n_ants=20, n_iter=60, alpha_aco=1.0, beta_aco=2.0, rho=0.1,
            seed=789):
    """ACO with node pheromones."""
    np.random.seed(seed)
    n = len(candidates)

    tau = np.ones(n) * 0.5
    tau_min, tau_max = 0.01, 5.0

    # Heuristic
    pax_sub = passengers[:min(5000, len(passengers))]
    eta = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(pax_sub - candidates[i], axis=1)
        nearby = np.sum(dists < 20 * env.scale)
        gamma = env.get_terrain_cost_at(candidates[i][0], candidates[i][1])
        eta[i] = (nearby + 1) / (C_FIXED + C_TERRAIN * gamma + 1)
    eta /= (eta.max() + 1e-10)

    best_cost = float('inf')
    best_indices = []

    t0 = time.time()
    for iteration in range(n_iter):
        iter_solutions = []
        for ant in range(n_ants):
            n_stops = np.random.randint(2, 8)
            selected = []
            available = list(range(n))
            for _ in range(n_stops):
                if not available:
                    break
                probs = np.array([
                    (tau[j] ** alpha_aco) * (eta[j] ** beta_aco)
                    for j in available])
                total_p = probs.sum()
                if total_p < 1e-15:
                    probs = np.ones(len(available)) / len(available)
                else:
                    probs = probs / total_p
                    probs = probs / probs.sum()
                chosen_local = np.random.choice(len(available), p=probs)
                chosen = available[chosen_local]
                selected.append(chosen)
                available.remove(chosen)

            cost = quick_cost(env, selected, candidates, dist_cache,
                              passengers, n_total)
            iter_solutions.append((selected, cost))

        iter_solutions.sort(key=lambda x: x[1])
        iter_best_idx, iter_best_cost = iter_solutions[0]

        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_indices = iter_best_idx[:]

        tau *= (1 - rho)
        deposit = 100.0 / (iter_best_cost + 1e-10)
        for idx in iter_best_idx:
            tau[idx] += deposit
        deposit_global = 100.0 / (best_cost + 1e-10) * 0.5
        for idx in best_indices:
            tau[idx] += deposit_global
        tau = np.clip(tau, tau_min, tau_max)

    elapsed = time.time() - t0
    result = full_evaluate(env, best_indices, candidates, dist_cache,
                           path_cache, passengers, n_total)
    return result, elapsed


# =============================================================================
# Experiment Configuration
# =============================================================================

SCALES = [
    # (label, grid_size, n_passengers, n_candidates, grid_res, algorithms)
    ('50',   100,   50,       12, 1,
     ['Exact', 'Two-Stage', 'GA', 'SA', 'ACO']),
    ('500',  200,   500,      15, 1,
     ['Exact', 'Two-Stage', 'GA', 'SA', 'ACO']),
    ('5K',   500,   5_000,    15, 2,
     ['Exact', 'Two-Stage', 'GA', 'SA', 'ACO']),
    ('50K',  1_000, 50_000,   20, 5,
     ['Two-Stage', 'GA', 'SA', 'ACO']),
    ('500K', 2_000, 500_000,  20, 10,
     ['Two-Stage', 'GA', 'SA', 'ACO']),
    ('2M',   5_000, 2_000_000, 20, 25,
     ['Two-Stage', 'GA', 'SA', 'ACO']),
    ('10M',  10_000, 10_000_000, 20, 50,
     ['Two-Stage', 'SA']),
]

# Algorithm parameter scaling
def algo_params(label):
    """Return scaled algorithm parameters based on problem size."""
    if label in ('50', '500', '5K'):
        return {'ga_gen': 80, 'sa_iter': 1500, 'aco_iter': 60}
    elif label in ('50K', '500K'):
        return {'ga_gen': 50, 'sa_iter': 800, 'aco_iter': 40}
    elif label == '2M':
        return {'ga_gen': 30, 'sa_iter': 500, 'aco_iter': 25}
    else:  # 10M
        return {'ga_gen': 30, 'sa_iter': 300, 'aco_iter': 25}


# =============================================================================
# Figure Generation
# =============================================================================

def plot_scale_time(all_results):
    """Fig 1: Log-log — passengers vs. computation time per algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    algo_colors = {
        'Exact': '#2196F3', 'Two-Stage': '#FF9800', 'GA': '#4CAF50',
        'SA': '#F44336', 'ACO': '#9C27B0',
    }
    algo_markers = {
        'Exact': 's', 'Two-Stage': 'o', 'GA': '^', 'SA': 'D', 'ACO': 'v',
    }

    # Collect data per algorithm
    algo_data = defaultdict(lambda: ([], []))
    for label, grid, n_pax, _, _, algos in SCALES:
        if label not in all_results:
            continue
        for algo_name in algos:
            if algo_name in all_results[label]:
                algo_data[algo_name][0].append(n_pax)
                algo_data[algo_name][1].append(
                    all_results[label][algo_name]['time'])

    for algo_name, (xs, ys) in algo_data.items():
        ax.plot(xs, ys, '-' + algo_markers.get(algo_name, 'o'),
                color=algo_colors.get(algo_name, 'gray'),
                label=algo_name, lw=2, ms=8, alpha=0.85)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Passengers', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Scalability: Computation Time vs. Problem Size',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = f'{OUT}/fig_scale_time.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


def plot_scale_cost_per_pax(all_results):
    """Fig 2: Log x — passengers vs. total cost / passenger."""
    fig, ax = plt.subplots(figsize=(10, 6))
    algo_colors = {
        'Exact': '#2196F3', 'Two-Stage': '#FF9800', 'GA': '#4CAF50',
        'SA': '#F44336', 'ACO': '#9C27B0',
    }
    algo_markers = {
        'Exact': 's', 'Two-Stage': 'o', 'GA': '^', 'SA': 'D', 'ACO': 'v',
    }

    algo_data = defaultdict(lambda: ([], []))
    for label, grid, n_pax, _, _, algos in SCALES:
        if label not in all_results:
            continue
        for algo_name in algos:
            if algo_name in all_results[label]:
                total = all_results[label][algo_name]['result']['total']
                algo_data[algo_name][0].append(n_pax)
                algo_data[algo_name][1].append(total / n_pax)

    for algo_name, (xs, ys) in algo_data.items():
        ax.plot(xs, ys, '-' + algo_markers.get(algo_name, 'o'),
                color=algo_colors.get(algo_name, 'gray'),
                label=algo_name, lw=2, ms=8, alpha=0.85)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Passengers', fontsize=12)
    ax.set_ylabel('Total Cost / Passenger', fontsize=12)
    ax.set_title('Efficiency at Scale: Cost per Passenger',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = f'{OUT}/fig_scale_cost_per_pax.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


def plot_scale_routes(all_results, envs):
    """Fig 3: 2×2 route maps at 4 scales."""
    display_scales = ['50', '5K', '500K', '10M']
    # Pick best algorithm available at each scale
    best_algo_pref = ['Two-Stage', 'SA', 'GA', 'ACO', 'Exact']

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for ax_idx, label in enumerate(display_scales):
        row, col = divmod(ax_idx, 2)
        ax = axes[row][col]

        if label not in all_results or label not in envs:
            ax.set_title(f'{label} passengers — not available')
            ax.axis('off')
            continue

        env = envs[label]
        # Find best result
        result = None
        algo_used = None
        for algo in best_algo_pref:
            if algo in all_results[label]:
                result = all_results[label][algo]['result']
                algo_used = algo
                break

        if result is None:
            ax.set_title(f'{label} — no result')
            ax.axis('off')
            continue

        # Terrain (low-res)
        tgrid, step = env.get_terrain_grid_lowres(200)
        ax.imshow(tgrid, origin='lower',
                  extent=[0, env.grid_size, 0, env.grid_size],
                  cmap='YlOrBr', alpha=0.25, vmin=1.0, vmax=4.0)

        # Restricted zones
        for zone in env.restricted_zones:
            if zone['type'] == 'rect':
                ax.add_patch(plt.Rectangle(
                    (zone['x'], zone['y']), zone['w'], zone['h'],
                    facecolor='red', alpha=0.2, edgecolor='darkred',
                    lw=1, ls='--'))
            elif zone['type'] == 'circle':
                ax.add_patch(plt.Circle(
                    (zone['cx'], zone['cy']), zone['r'],
                    facecolor='red', alpha=0.2, edgecolor='darkred',
                    lw=1, ls='--'))

        # Subsample passengers for display
        pax = env.passengers
        if len(pax) > 2000:
            idx = np.random.choice(len(pax), 2000, replace=False)
            pax_show = pax[idx]
        else:
            pax_show = pax

        ax.scatter(pax_show[:, 0], pax_show[:, 1], c='limegreen',
                   s=max(2, 40 - ax_idx * 10), edgecolors='darkgreen',
                   lw=0.3, zorder=7, alpha=0.6)

        # Route
        route = result['route']
        if route is not None and len(route) > 1:
            ax.plot(route[:, 0], route[:, 1], 'b-', lw=2, zorder=5)

        # Stops
        stops = result['stops']
        if len(stops) > 0:
            ax.scatter(stops[:, 0], stops[:, 1], c='orange', s=100,
                       marker='s', edgecolors='darkorange', lw=1.5,
                       zorder=8)

        # A and B
        ax.scatter(*env.A, c='lime', s=200, marker='*',
                   edgecolors='black', lw=1, zorder=10)
        ax.scatter(*env.B, c='crimson', s=200, marker='*',
                   edgecolors='black', lw=1, zorder=10)

        n_pax_actual = env.n_passengers
        ax.set_title(
            f'{label} passengers ({env.grid_size}x{env.grid_size})\n'
            f'{algo_used}: {result["n_stops"]} stops, '
            f'cost={result["total"]:.0f}',
            fontsize=11, fontweight='bold')
        ax.set_xlim(-env.grid_size * 0.02, env.grid_size * 1.02)
        ax.set_ylim(-env.grid_size * 0.02, env.grid_size * 1.02)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)

    fig.suptitle('Route Maps at Different Scales',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = f'{OUT}/fig_scale_routes.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


def plot_scale_stops(all_results):
    """Fig 4: Log x — passengers vs. number of stops."""
    fig, ax = plt.subplots(figsize=(10, 6))
    algo_colors = {
        'Exact': '#2196F3', 'Two-Stage': '#FF9800', 'GA': '#4CAF50',
        'SA': '#F44336', 'ACO': '#9C27B0',
    }
    algo_markers = {
        'Exact': 's', 'Two-Stage': 'o', 'GA': '^', 'SA': 'D', 'ACO': 'v',
    }

    algo_data = defaultdict(lambda: ([], []))
    for label, grid, n_pax, _, _, algos in SCALES:
        if label not in all_results:
            continue
        for algo_name in algos:
            if algo_name in all_results[label]:
                n_stops = all_results[label][algo_name]['result']['n_stops']
                algo_data[algo_name][0].append(n_pax)
                algo_data[algo_name][1].append(n_stops)

    for algo_name, (xs, ys) in algo_data.items():
        ax.plot(xs, ys, '-' + algo_markers.get(algo_name, 'o'),
                color=algo_colors.get(algo_name, 'gray'),
                label=algo_name, lw=2, ms=8, alpha=0.85)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Passengers', fontsize=12)
    ax.set_ylabel('Number of Bus Stops', fontsize=12)
    ax.set_title('Optimal Stop Count vs. Problem Size',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    path = f'{OUT}/fig_scale_stops.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 72)
    print("  MULTI-SCALE EXPERIMENT: 50 -> 10M PASSENGERS")
    print(f"  Weights: alpha={ALPHA}, beta={BETA}, delta={DELTA}")
    print("=" * 72)

    all_results = {}   # label -> {algo_name -> {'result': ..., 'time': ...}}
    envs = {}          # label -> env

    for label, grid_size, n_pax, n_cand, grid_res, algos in SCALES:
        print(f"\n{'='*72}")
        print(f"  SCALE: {label}  |  Grid: {grid_size}x{grid_size}  |  "
              f"Passengers: {n_pax:,}  |  grid_res: {grid_res}")
        print(f"  Algorithms: {', '.join(algos)}")
        print(f"{'='*72}")

        # Create environment
        t_env = time.time()
        env = ScalableEnvironment(grid_size=grid_size,
                                  n_passengers=n_pax, seed=42)
        envs[label] = env
        print(f"  Environment created in {time.time()-t_env:.1f}s "
              f"({len(env.passengers):,} passengers)")

        # Passenger sample for optimization
        if n_pax > PAX_SAMPLE_CAP:
            idx = np.random.choice(n_pax, PAX_SAMPLE_CAP, replace=False)
            pax_sample = env.passengers[idx]
            print(f"  Using {PAX_SAMPLE_CAP:,} passenger sample "
                  f"for optimization")
        else:
            pax_sample = env.passengers

        # Generate candidates
        t_cand = time.time()
        candidates = generate_candidates(env, target_n=n_cand,
                                         pax_for_cluster=pax_sample)
        print(f"  {len(candidates)} candidates generated in "
              f"{time.time()-t_cand:.1f}s")

        # Build distance cache
        t_cache = time.time()
        dist_cache, path_cache, _ = build_distance_cache(
            env, candidates, grid_res)
        print(f"  Distance cache built in {time.time()-t_cache:.1f}s")

        params = algo_params(label)
        scale_results = {}

        # Run each algorithm
        for algo_name in algos:
            print(f"\n  --- {algo_name} ---")
            np.random.seed(42)  # reset for reproducibility

            if algo_name == 'Exact':
                result, elapsed = run_exact(
                    env, candidates, dist_cache, path_cache,
                    pax_sample, n_pax)
            elif algo_name == 'Two-Stage':
                result, elapsed = run_two_stage(
                    env, pax_sample, n_pax, grid_res)
            elif algo_name == 'GA':
                result, elapsed = run_ga(
                    env, candidates, dist_cache, path_cache,
                    pax_sample, n_pax,
                    n_gen=params['ga_gen'])
            elif algo_name == 'SA':
                result, elapsed = run_sa(
                    env, candidates, dist_cache, path_cache,
                    pax_sample, n_pax,
                    n_iter=params['sa_iter'])
            elif algo_name == 'ACO':
                result, elapsed = run_aco(
                    env, candidates, dist_cache, path_cache,
                    pax_sample, n_pax,
                    n_iter=params['aco_iter'])
            else:
                continue

            scale_results[algo_name] = {'result': result, 'time': elapsed}
            print(f"  {algo_name}: cost={result['total']:.1f}, "
                  f"stops={result['n_stops']}, time={elapsed:.2f}s")

        all_results[label] = scale_results

    # ── Generate Figures ──
    print(f"\n{'='*72}")
    print("  GENERATING FIGURES")
    print(f"{'='*72}")

    plot_scale_time(all_results)
    plot_scale_cost_per_pax(all_results)
    plot_scale_routes(all_results, envs)
    plot_scale_stops(all_results)

    # ── Summary Table ──
    print(f"\n{'='*72}")
    print("  MULTI-SCALE EXPERIMENT SUMMARY")
    print(f"{'='*72}")

    header = (f"  {'Scale':<8} {'Algo':<12} {'Total':>10} {'Route':>10} "
              f"{'Station':>10} {'Walking':>10} {'Stops':>6} "
              f"{'Time(s)':>8} {'Cost/Pax':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, grid, n_pax, _, _, algos in SCALES:
        if label not in all_results:
            continue
        for algo_name in algos:
            if algo_name not in all_results[label]:
                continue
            r = all_results[label][algo_name]
            res = r['result']
            print(f"  {label:<8} {algo_name:<12} "
                  f"{res['total']:>10.1f} "
                  f"{res.get('route_cost', res.get('route', 0)):>10.1f} "
                  f"{res.get('station_cost', res.get('station', 0)):>10.1f} "
                  f"{res.get('walking_cost', res.get('walking', 0)):>10.1f} "
                  f"{res['n_stops']:>6} "
                  f"{r['time']:>8.2f} "
                  f"{res['total']/n_pax:>10.4f}")

    print(f"\n  Generated 4 figures:")
    print(f"    fig_scale_time.png")
    print(f"    fig_scale_cost_per_pax.png")
    print(f"    fig_scale_routes.png")
    print(f"    fig_scale_stops.png")
    print(f"\nDone!")
