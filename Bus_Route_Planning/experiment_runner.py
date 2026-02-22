"""
Bus Route Optimization — 5-Algorithm Experiment Runner
======================================================
Implements: Exact Enumeration, Two-Stage Heuristic, GA, SA, ACO
Generates: 12 publication-ready PNG figures
"""

import sys
import os
import time
import numpy as np
from itertools import combinations
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Import shared infrastructure from existing optimizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bus_route_optimizer import (
    Environment, Solution, ALPHA, BETA, DELTA, C_FIXED, C_TERRAIN,
    astar, build_route, assign_passengers, compute_costs,
    snap_to_valid, order_stops_along_AB, op_smooth_route,
    op_two_opt_ordering, op_merge_nearby,
    run_optimization_loop, plot_final_solution,
)
from sklearn.cluster import KMeans

OUT = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# A. Shared Helpers
# =============================================================================

def generate_candidate_stops(env, target_n=12):
    """Generate candidate stops via KMeans union + passenger-density points."""
    all_centers = []
    for k in [3, 5, 7, 9]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(env.passengers)
        all_centers.extend(km.cluster_centers_)

    # Also add density-based candidates: passengers far from A and B
    dists_to_AB = np.minimum(
        np.linalg.norm(env.passengers - env.A, axis=1),
        np.linalg.norm(env.passengers - env.B, axis=1))
    far_idx = np.argsort(dists_to_AB)[-6:]
    all_centers.extend(env.passengers[far_idx])

    snapped = []
    for c in all_centers:
        s = snap_to_valid(env, c)
        if s is not None:
            if all(np.linalg.norm(s - ex) >= 5 for ex in snapped):
                snapped.append(s)

    candidates = np.array(snapped[:target_n])
    print(f"  Generated {len(candidates)} candidate stops")
    return candidates


def build_distance_cache(env, candidates):
    """Pre-compute A* distances and smoothed paths between all waypoint pairs.
    Waypoints: A (idx=0), candidates[0..n-1] (idx=1..n), B (idx=n+1)
    """
    waypoints = [env.A] + list(candidates) + [env.B]
    n = len(waypoints)
    dist_cache = np.full((n, n), np.inf)
    path_cache = {}

    total_pairs = n * (n - 1) // 2
    print(f"  Pre-computing A* for {n} waypoints ({total_pairs} pairs)...")
    done = 0
    for i in range(n):
        dist_cache[i][i] = 0.0
        for j in range(i + 1, n):
            path = astar(env, waypoints[i], waypoints[j], grid_res=1)
            if path is not None:
                smooth = op_smooth_route(np.array(path), env)
                d = sum(np.linalg.norm(smooth[k + 1] - smooth[k])
                        for k in range(len(smooth) - 1))
                dist_cache[i][j] = d
                dist_cache[j][i] = d
                path_cache[(i, j)] = smooth
                path_cache[(j, i)] = smooth[::-1]
            else:
                d = np.linalg.norm(waypoints[i] - waypoints[j])
                dist_cache[i][j] = d
                dist_cache[j][i] = d
                path_cache[(i, j)] = np.array([waypoints[i], waypoints[j]])
                path_cache[(j, i)] = np.array([waypoints[j], waypoints[i]])
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{total_pairs} pairs computed")
    print(f"  Distance cache complete.")
    return dist_cache, path_cache, waypoints


def quick_cost(env, selected_indices, candidates, dist_cache):
    """Fast cost evaluation using cached A* distances. Returns total cost."""
    n_cand = len(candidates)
    stops = candidates[selected_indices] if len(selected_indices) > 0 else np.empty((0, 2))

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

    # Walking cost
    if len(stops) > 0:
        all_s = np.vstack([env.A] + list(stops) + [env.B])
    else:
        all_s = np.vstack([env.A, env.B])
    walk_total = 0.0
    for p in env.passengers:
        dists = np.linalg.norm(all_s - p, axis=1)
        walk_total += np.min(dists)
    walking_cost = DELTA * walk_total

    return route_cost + station_cost + walking_cost


def full_evaluate(env, selected_indices, candidates, dist_cache, path_cache):
    """Full evaluation returning a Solution with reconstructed route."""
    n_cand = len(candidates)
    stops = candidates[selected_indices] if len(selected_indices) > 0 else np.empty((0, 2))

    sol = Solution(env, stops)

    if len(stops) == 0:
        route = path_cache.get((0, n_cand + 1), np.array([env.A, env.B]))
        sol.route = route
    else:
        ab = env.B - env.A
        ab_n = ab / (np.linalg.norm(ab) + 1e-9)
        projs = [(np.dot(stops[i] - env.A, ab_n), selected_indices[i], i)
                 for i in range(len(stops))]
        projs.sort()
        cache_wp = [0] + [p[1] + 1 for p in projs] + [n_cand + 1]
        ordered_stop_idx = [p[2] for p in projs]
        sol.stops = stops[ordered_stop_idx]

        segments = []
        for i in range(len(cache_wp) - 1):
            seg = path_cache.get((cache_wp[i], cache_wp[i + 1]))
            if seg is None:
                all_wp = [env.A] + list(candidates) + [env.B]
                seg = np.array([all_wp[cache_wp[i]], all_wp[cache_wp[i + 1]]])
            if i > 0 and len(segments) > 0:
                seg = seg[1:]
            segments.append(seg)
        sol.route = np.vstack(segments) if segments else np.array([env.A, env.B])

    sol.assignments, sol.all_stops = assign_passengers(env, sol.stops)
    sol.costs = compute_costs(env, sol.stops, sol.route,
                              sol.assignments, sol.all_stops)
    return sol


# =============================================================================
# B. Algorithm Implementations
# =============================================================================

# ── 1. Exact Enumeration ─────────────────────────────────────────────────────

def run_exact(env, candidates, dist_cache, path_cache, max_stops=8):
    """Enumerate all C(n, 1..max_stops) subsets."""
    print("\n" + "=" * 60)
    print("  ALGORITHM 1: EXACT ENUMERATION")
    print("=" * 60)

    n = len(candidates)
    best_cost = float('inf')
    best_indices = []
    history = []
    eval_count = 0

    t0 = time.time()
    for k in range(1, min(max_stops, n) + 1):
        for combo in combinations(range(n), k):
            indices = list(combo)
            cost = quick_cost(env, indices, candidates, dist_cache)
            eval_count += 1

            if cost < best_cost:
                best_cost = cost
                best_indices = indices
                history.append((eval_count, best_cost))

            if eval_count % 500 == 0:
                print(f"    {eval_count} subsets evaluated, best: {best_cost:.1f}")

    elapsed = time.time() - t0
    if not history:
        history.append((eval_count, best_cost))
    print(f"  Exact: {eval_count} evals in {elapsed:.1f}s, best={best_cost:.1f} "
          f"({len(best_indices)} stops)")

    sol = full_evaluate(env, best_indices, candidates, dist_cache, path_cache)
    return sol, history, elapsed, eval_count


# ── 2. Two-Stage Heuristic ───────────────────────────────────────────────────

def run_two_stage(env):
    """Wrap existing 10-round iterative heuristic."""
    print("\n" + "=" * 60)
    print("  ALGORITHM 2: TWO-STAGE HEURISTIC (10-round)")
    print("=" * 60)

    t0 = time.time()
    history_raw, reviews, solvers = run_optimization_loop(env)
    elapsed = time.time() - t0

    history = [(i + 1, h['total']) for i, h in enumerate(history_raw)]
    best_sol = solvers[-1]
    print(f"  Two-Stage: 10 rounds in {elapsed:.1f}s, "
          f"best={best_sol.costs['total']:.1f} ({best_sol.costs['n_stops']} stops)")
    return best_sol, history, elapsed, 10


# ── 3. Genetic Algorithm ─────────────────────────────────────────────────────

def run_ga(env, candidates, dist_cache, path_cache,
           pop_size=40, n_gen=80, p_cross=0.8, p_mut=0.15, seed=123):
    """Binary-encoded GA with tournament selection, uniform crossover, bit-flip."""
    print("\n" + "=" * 60)
    print("  ALGORITHM 3: GENETIC ALGORITHM")
    print("=" * 60)

    np.random.seed(seed)
    n = len(candidates)

    # ── Initialise population ──
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
        return quick_cost(env, idx, candidates, dist_cache)

    costs = [fitness(c) for c in population]
    best_cost = min(costs)
    best_chromo = population[int(np.argmin(costs))].copy()
    history = [(pop_size, best_cost)]
    eval_count = pop_size

    t0 = time.time()
    for gen in range(n_gen):
        new_pop = []
        ranked = np.argsort(costs)
        new_pop.append(population[ranked[0]].copy())
        new_pop.append(population[ranked[1]].copy())

        while len(new_pop) < pop_size:
            # Tournament (k=3)
            def tourney():
                idx = np.random.choice(pop_size, 3, replace=False)
                return population[idx[np.argmin([costs[i] for i in idx])]].copy()

            p1, p2 = tourney(), tourney()

            # Uniform crossover
            if np.random.random() < p_cross:
                mask = np.random.randint(0, 2, n)
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Bit-flip mutation
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
        eval_count += pop_size

        gen_best = min(costs)
        if gen_best < best_cost:
            best_cost = gen_best
            best_chromo = population[int(np.argmin(costs))].copy()

        history.append((eval_count, best_cost))
        if (gen + 1) % 20 == 0:
            print(f"    Gen {gen+1}/{n_gen}: best={best_cost:.1f}")

    elapsed = time.time() - t0
    best_idx = list(np.where(best_chromo == 1)[0])
    sol = full_evaluate(env, best_idx, candidates, dist_cache, path_cache)
    print(f"  GA: {eval_count} evals in {elapsed:.1f}s, "
          f"best={sol.costs['total']:.1f} ({sol.costs['n_stops']} stops)")
    return sol, history, elapsed, eval_count


# ── 4. Simulated Annealing ───────────────────────────────────────────────────

def run_sa(env, candidates, dist_cache, path_cache,
           T0=200, cooling=0.995, n_iter=1500, seed=456):
    """SA with binary vector, flip/swap neighbours."""
    print("\n" + "=" * 60)
    print("  ALGORITHM 4: SIMULATED ANNEALING")
    print("=" * 60)

    np.random.seed(seed)
    n = len(candidates)

    # Initial solution — random 3-5 stops
    current = np.zeros(n, dtype=int)
    k = np.random.randint(3, 6)
    current[np.random.choice(n, k, replace=False)] = 1

    current_cost = quick_cost(env, list(np.where(current == 1)[0]),
                              candidates, dist_cache)
    best = current.copy()
    best_cost = current_cost

    T = T0
    history = [(1, best_cost)]
    eval_count = 1

    t0 = time.time()
    for it in range(n_iter):
        neighbor = current.copy()
        move = np.random.random()

        if move < 0.35:
            # Flip one bit
            neighbor[np.random.randint(n)] ^= 1
        elif move < 0.65:
            # Swap: turn off one, turn on another
            ones = np.where(neighbor == 1)[0]
            zeros = np.where(neighbor == 0)[0]
            if len(ones) > 0 and len(zeros) > 0:
                neighbor[np.random.choice(ones)] = 0
                neighbor[np.random.choice(zeros)] = 1
        else:
            # Flip two bits
            i1, i2 = np.random.choice(n, 2, replace=False)
            neighbor[i1] ^= 1
            neighbor[i2] ^= 1

        # Enforce constraints
        if neighbor.sum() == 0:
            neighbor[np.random.randint(n)] = 1
        while neighbor.sum() > 8:
            ones = np.where(neighbor == 1)[0]
            neighbor[np.random.choice(ones)] = 0

        idx = list(np.where(neighbor == 1)[0])
        neighbor_cost = quick_cost(env, idx, candidates, dist_cache)
        eval_count += 1

        delta = neighbor_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost

        T *= cooling
        history.append((eval_count, best_cost))

        if (it + 1) % 300 == 0:
            print(f"    Iter {it+1}/{n_iter}: T={T:.2f}, "
                  f"current={current_cost:.1f}, best={best_cost:.1f}")

    elapsed = time.time() - t0
    best_idx = list(np.where(best == 1)[0])
    sol = full_evaluate(env, best_idx, candidates, dist_cache, path_cache)
    print(f"  SA: {eval_count} evals in {elapsed:.1f}s, "
          f"best={sol.costs['total']:.1f} ({sol.costs['n_stops']} stops)")
    return sol, history, elapsed, eval_count


# ── 5. Ant Colony Optimization ───────────────────────────────────────────────

def run_aco(env, candidates, dist_cache, path_cache,
            n_ants=20, n_iter=60, alpha_aco=1.0, beta_aco=2.0,
            rho=0.1, seed=789):
    """ACO with node pheromones for stop selection (MMAS variant)."""
    print("\n" + "=" * 60)
    print("  ALGORITHM 5: ANT COLONY OPTIMIZATION")
    print("=" * 60)

    np.random.seed(seed)
    n = len(candidates)

    # Pheromones
    tau = np.ones(n) * 0.5
    tau_min, tau_max = 0.01, 5.0

    # Heuristic: passenger-density / station-cost
    eta = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(env.passengers - candidates[i], axis=1)
        nearby = np.sum(dists < 20)
        gamma = env.get_terrain_cost_at(candidates[i][0], candidates[i][1])
        eta[i] = (nearby + 1) / (C_FIXED + C_TERRAIN * gamma + 1)
    eta /= (eta.max() + 1e-10)

    best_cost = float('inf')
    best_indices = []
    history = [(0, float('inf'))]
    eval_count = 0

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
                    for j in available
                ])
                total_p = probs.sum()
                if total_p < 1e-15:
                    probs = np.ones(len(available)) / len(available)
                else:
                    probs = probs / total_p
                    probs = probs / probs.sum()  # ensure exact sum=1
                chosen_local = np.random.choice(len(available), p=probs)
                chosen = available[chosen_local]
                selected.append(chosen)
                available.remove(chosen)

            cost = quick_cost(env, selected, candidates, dist_cache)
            eval_count += 1
            iter_solutions.append((selected, cost))

        iter_solutions.sort(key=lambda x: x[1])
        iter_best_idx, iter_best_cost = iter_solutions[0]

        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_indices = iter_best_idx[:]

        # Pheromone update
        tau *= (1 - rho)
        deposit = 100.0 / (iter_best_cost + 1e-10)
        for idx in iter_best_idx:
            tau[idx] += deposit
        deposit_global = 100.0 / (best_cost + 1e-10) * 0.5
        for idx in best_indices:
            tau[idx] += deposit_global
        tau = np.clip(tau, tau_min, tau_max)

        history.append((eval_count, best_cost))
        if (iteration + 1) % 15 == 0:
            print(f"    Iter {iteration+1}/{n_iter}: best={best_cost:.1f}, "
                  f"iter_best={iter_best_cost:.1f}")

    elapsed = time.time() - t0
    sol = full_evaluate(env, best_indices, candidates, dist_cache, path_cache)
    print(f"  ACO: {eval_count} evals in {elapsed:.1f}s, "
          f"best={sol.costs['total']:.1f} ({sol.costs['n_stops']} stops)")
    return sol, history, elapsed, eval_count


# =============================================================================
# C. Figure Generation
# =============================================================================

def _draw_env(ax, env):
    """Draw terrain + restricted zones on an axes."""
    ax.imshow(env.terrain_cost, origin='lower', extent=[0, 100, 0, 100],
              cmap='YlOrBr', alpha=0.25, vmin=1.0, vmax=4.0)
    for zone in env.restricted_zones:
        if zone['type'] == 'rect':
            ax.add_patch(plt.Rectangle(
                (zone['x'], zone['y']), zone['w'], zone['h'],
                facecolor='red', alpha=0.2, edgecolor='darkred', lw=1.5, ls='--'))
        elif zone['type'] == 'circle':
            ax.add_patch(plt.Circle(
                (zone['cx'], zone['cy']), zone['r'],
                facecolor='red', alpha=0.2, edgecolor='darkred', lw=1.5, ls='--'))


def _draw_solution(ax, env, sol, show_assignments=True):
    """Draw passengers, stops, route, A/B on axes."""
    if show_assignments and sol.assignments is not None:
        for i, p in enumerate(env.passengers):
            stop = sol.all_stops[sol.assignments[i]]
            ax.plot([p[0], stop[0]], [p[1], stop[1]],
                    color='gray', ls='--', alpha=0.2, lw=0.6)

    if sol.route is not None and len(sol.route) > 1:
        ax.plot(sol.route[:, 0], sol.route[:, 1], 'b-', lw=2.5,
                label='Bus Route', zorder=5)
        n_arr = min(8, len(sol.route) - 1)
        for idx in np.linspace(0, len(sol.route) - 2, n_arr, dtype=int):
            ax.annotate('', xy=sol.route[idx + 1], xytext=sol.route[idx],
                        arrowprops=dict(arrowstyle='->', color='royalblue',
                                        lw=1.2),
                        zorder=6)

    ax.scatter(env.passengers[:, 0], env.passengers[:, 1], c='limegreen',
               s=40, edgecolors='darkgreen', lw=0.5, zorder=7,
               label=f'Passengers (n={len(env.passengers)})')

    if len(sol.stops) > 0:
        ax.scatter(sol.stops[:, 0], sol.stops[:, 1], c='orange', s=180,
                   marker='s', edgecolors='darkorange', lw=2, zorder=8,
                   label=f'Bus Stops (n={len(sol.stops)})')
        for idx, s in enumerate(sol.stops):
            ax.annotate(f'S{idx+1}', s, fontsize=8, fontweight='bold',
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', zorder=9)

    ax.scatter(*env.A, c='lime', s=300, marker='*', edgecolors='black',
               lw=1.5, zorder=10, label='A (Start)')
    ax.scatter(*env.B, c='crimson', s=300, marker='*', edgecolors='black',
               lw=1.5, zorder=10, label='B (End)')
    ax.annotate('A', env.A, fontsize=12, fontweight='bold',
                xytext=(6, 6), textcoords='offset points')
    ax.annotate('B', env.B, fontsize=12, fontweight='bold',
                xytext=(6, 6), textcoords='offset points')


def plot_route_map(env, sol, title, save_path):
    """Individual route map figure."""
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_env(ax, env)
    _draw_solution(ax, env, sol)

    c = sol.costs
    info = (f"Total: {c['total']:.1f}\n"
            f"Route: {c['route']:.1f} (len={c['route_length']:.0f})\n"
            f"Station: {c['station']:.1f} ({c['n_stops']} stops)\n"
            f"Walking: {c['walking']:.1f} (avg={c['avg_walk']:.1f})")
    ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=9,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      alpha=0.9))

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_convergence_curve(history, title, save_path, color='blue'):
    """Convergence curve for one algorithm."""
    # Filter out inf values
    filtered = [(e, c) for e, c in history if np.isfinite(c)]
    if not filtered:
        print(f"  [Skip] {save_path} — no finite data")
        return
    evals = [h[0] for h in filtered]
    costs = [h[1] for h in filtered]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(evals, costs, '-', color=color, lw=2, marker='.', markersize=4)
    ax.fill_between(evals, costs, alpha=0.08, color=color)

    ax.set_xlabel('Evaluations / Rounds', fontsize=12)
    ax.set_ylabel('Best Total Cost', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Text annotation instead of arrow (avoids matplotlib arrow bugs)
    ax.text(evals[-1], costs[-1], f'  {costs[-1]:.1f}',
            fontsize=11, fontweight='bold', color=color, va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_comparison_bars(results, save_cost, save_time):
    """Stacked cost bars + computation time bars."""
    names = list(results.keys())
    totals = [results[n]['sol'].costs['total'] for n in names]
    routes = [results[n]['sol'].costs['route'] for n in names]
    stations = [results[n]['sol'].costs['station'] for n in names]
    walkings = [results[n]['sol'].costs['walking'] for n in names]
    times = [results[n]['time'] for n in names]

    # ── Cost comparison (stacked bar) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    w = 0.55

    ax.bar(x, routes, w, label='Route Cost', color='steelblue', alpha=0.85)
    ax.bar(x, stations, w, bottom=routes, label='Station Cost',
           color='darkorange', alpha=0.85)
    bottoms = [r + s for r, s in zip(routes, stations)]
    ax.bar(x, walkings, w, bottom=bottoms, label='Walking Cost',
           color='forestgreen', alpha=0.85)

    for i, t in enumerate(totals):
        ax.text(i, t + 5, f'{t:.0f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title(
        'Cost Breakdown: 5-Algorithm Comparison\n'
        f'($\\alpha$={ALPHA}, $\\beta$={BETA}, $\\delta$={DELTA})',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_cost, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_cost}")

    # ── Time comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']
    ax.bar(x, times, w, color=colors, alpha=0.85)
    for i, t in enumerate(times):
        ax.text(i, t + max(times) * 0.02, f'{t:.1f}s', ha='center',
                va='bottom', fontweight='bold', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Computation Time Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(save_time, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_time}")


def plot_combined_routes(env, results, save_path):
    """2x3 grid: 5 route maps + summary panel."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.25)
    algo_names = list(results.keys())

    for idx, name in enumerate(algo_names):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        sol = results[name]['sol']

        _draw_env(ax, env)

        if sol.route is not None and len(sol.route) > 1:
            ax.plot(sol.route[:, 0], sol.route[:, 1], 'b-', lw=2, zorder=5)
        ax.scatter(env.passengers[:, 0], env.passengers[:, 1], c='limegreen',
                   s=20, edgecolors='darkgreen', lw=0.3, zorder=7)
        if len(sol.stops) > 0:
            ax.scatter(sol.stops[:, 0], sol.stops[:, 1], c='orange', s=100,
                       marker='s', edgecolors='darkorange', lw=1.5, zorder=8)
        ax.scatter(*env.A, c='lime', s=200, marker='*', edgecolors='black',
                   lw=1, zorder=10)
        ax.scatter(*env.B, c='crimson', s=200, marker='*', edgecolors='black',
                   lw=1, zorder=10)

        c = sol.costs
        ax.set_title(f'{name}\nTotal={c["total"]:.0f}, Stops={c["n_stops"]}',
                     fontsize=11, fontweight='bold')
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)

    # 6th panel — summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    header = f"{'Algorithm':<12} {'Cost':>7} {'Stops':>6} {'Time':>7}\n"
    header += "-" * 36 + "\n"
    body = ""
    for name in algo_names:
        c = results[name]['sol'].costs
        t = results[name]['time']
        body += f"{name:<12} {c['total']:>7.0f} {c['n_stops']:>6} {t:>6.1f}s\n"
    ax.text(0.1, 0.90, header + body, transform=ax.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig.suptitle('Bus Route Optimization: 5-Algorithm Comparison',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 72)
    print("  BUS ROUTE OPTIMIZATION — 5-ALGORITHM EXPERIMENT")
    print(f"  Weights: alpha={ALPHA}, beta={BETA}, delta={DELTA}")
    print("=" * 72)

    # ── Environment ──
    env = Environment(grid_size=100, n_passengers=50, seed=42)
    print(f"\n  Environment: {env.grid_size}x{env.grid_size}, "
          f"{len(env.passengers)} passengers, "
          f"A=({env.A[0]:.0f},{env.A[1]:.0f}), B=({env.B[0]:.0f},{env.B[1]:.0f})")

    # ── Candidate stops ──
    print("\n[Step 1] Generating candidate stops...")
    candidates = generate_candidate_stops(env, target_n=12)

    # ── Distance cache ──
    print("\n[Step 2] Building A* distance cache...")
    dist_cache, path_cache, waypoints = build_distance_cache(env, candidates)

    # ── Run all 5 algorithms ──
    print("\n[Step 3] Running algorithms...")
    results = {}

    sol_exact, hist_exact, time_exact, evals_exact = run_exact(
        env, candidates, dist_cache, path_cache)
    results['Exact'] = {'sol': sol_exact, 'history': hist_exact,
                        'time': time_exact, 'evals': evals_exact}

    sol_ts, hist_ts, time_ts, evals_ts = run_two_stage(env)
    results['Two-Stage'] = {'sol': sol_ts, 'history': hist_ts,
                            'time': time_ts, 'evals': evals_ts}

    sol_ga, hist_ga, time_ga, evals_ga = run_ga(
        env, candidates, dist_cache, path_cache)
    results['GA'] = {'sol': sol_ga, 'history': hist_ga,
                     'time': time_ga, 'evals': evals_ga}

    sol_sa, hist_sa, time_sa, evals_sa = run_sa(
        env, candidates, dist_cache, path_cache)
    results['SA'] = {'sol': sol_sa, 'history': hist_sa,
                     'time': time_sa, 'evals': evals_sa}

    sol_aco, hist_aco, time_aco, evals_aco = run_aco(
        env, candidates, dist_cache, path_cache)
    results['ACO'] = {'sol': sol_aco, 'history': hist_aco,
                      'time': time_aco, 'evals': evals_aco}

    # ── Generate 12 figures ──
    print("\n" + "=" * 72)
    print("  GENERATING FIGURES")
    print("=" * 72)

    # 5 route maps
    for name in results:
        fname = name.lower().replace('-', '_')
        plot_route_map(env, results[name]['sol'],
                       f'Bus Route — {name}',
                       f'{OUT}/fig_route_{fname}.png')

    # 4 convergence curves
    conv_colors = {'Two-Stage': 'orange', 'GA': 'green',
                   'SA': 'red', 'ACO': 'purple'}
    for name, color in conv_colors.items():
        fname = name.lower().replace('-', '_')
        plot_convergence_curve(results[name]['history'],
                               f'Convergence — {name}',
                               f'{OUT}/fig_conv_{fname}.png',
                               color=color)

    # Cost + time bars
    plot_comparison_bars(results,
                         f'{OUT}/fig_comparison_costs.png',
                         f'{OUT}/fig_comparison_time.png')

    # Combined 2x3 grid
    plot_combined_routes(env, results, f'{OUT}/fig_combined_routes.png')

    # ── Summary table ──
    print("\n" + "=" * 72)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 72)

    exact_cost = results['Exact']['sol'].costs['total']

    print(f"\n  {'Algorithm':<12} {'Total':>8} {'Route':>8} {'Station':>8} "
          f"{'Walking':>8} {'#Stops':>7} {'AvgWalk':>8} {'MaxWalk':>8} "
          f"{'Time':>7} {'Evals':>7} {'Gap%':>7}")
    print("  " + "-" * 104)

    for name in results:
        c = results[name]['sol'].costs
        t = results[name]['time']
        e = results[name]['evals']
        gap = (c['total'] - exact_cost) / exact_cost * 100
        print(f"  {name:<12} {c['total']:>8.1f} {c['route']:>8.1f} "
              f"{c['station']:>8.1f} {c['walking']:>8.1f} "
              f"{c['n_stops']:>7} {c['avg_walk']:>8.1f} {c['max_walk']:>8.1f} "
              f"{t:>6.1f}s {e:>7} {gap:>6.1f}%")

    print(f"\n  Exact baseline cost: {exact_cost:.1f}")

    print("\n  Generated figures:")
    for name in results:
        fname = name.lower().replace('-', '_')
        print(f"    fig_route_{fname}.png")
    for name in conv_colors:
        fname = name.lower().replace('-', '_')
        print(f"    fig_conv_{fname}.png")
    print("    fig_comparison_costs.png")
    print("    fig_comparison_time.png")
    print("    fig_combined_routes.png")
    print(f"\n  Total: 12 figures")
    print("\nDone!")
