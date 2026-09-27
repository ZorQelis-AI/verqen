"""Game and navigation decision families (answers from search / game-theoretic solutions)."""

from __future__ import annotations

import heapq
import random
from collections import deque
from functools import lru_cache

from .common import CITIES, choice_q, example, noul_q, numeric_options, pick

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _bfs(grid, start, goal):
    H, W = len(grid), len(grid[0])
    dist = {goal: 0}
    dq = deque([goal])
    while dq:
        r, c = dq.popleft()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
    return dist


def grid_nav(rng: random.Random, variant: int):
    for _ in range(50):
        H, W = rng.randint(4, 8), rng.randint(4, 8)
        grid = [["#" if rng.random() < rng.uniform(0.1, 0.3) else "." for _ in range(W)] for _ in range(H)]
        cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == "."]
        if len(cells) < 4:
            continue
        a, g = rng.sample(cells, 2)
        dist = _bfs(grid, a, g)
        if a not in dist or dist[a] < 1:
            continue
        break
    else:
        return None
    grid[a[0]][a[1]] = "A"
    grid[g[0]][g[1]] = "G"
    opts = list(MOVES)
    good = set()
    for m, (dr, dc) in MOVES.items():
        n = (a[0] + dr, a[1] + dc)
        if n in dist and dist[n] == dist[a] - 1:
            good.add(m)
    if variant % 2 == 0:
        board = "\n".join(" ".join(row) for row in grid)
        state = f"Grid map ({H} rows x {W} cols). A = agent, G = goal, # = wall, . = open. Row 0 is the top.\n{board}"
    else:
        walls = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == "#"]
        state = (f"A {H}x{W} maze uses (row, col) coordinates with row 0 at the top. The agent stands at {a}, "
                 f"the exit is at {g}. Blocked cells: {walls}.")
    qtext = pick(rng, ["Which move brings the agent closest to the goal along a shortest path?",
                       "What is the best next move?", "Choose the next step on an optimal route to the goal."])
    qs = [choice_q(qtext, opts, [1.0 if o in good else 0.0 for o in opts], rng)]
    if rng.random() < 0.5:
        d = dist[a]
        levels = ["1-2 steps", "3-4 steps", "5-7 steps", "8-11 steps", "12 or more steps"]
        lvl = 0 if d <= 2 else 1 if d <= 4 else 2 if d <= 7 else 3 if d <= 11 else 4
        qs.append(choice_q("How long is the shortest path to the goal?", levels, lvl, rng, qtype="score"))
    if rng.random() < 0.3:
        m = pick(rng, opts)
        dr, dc = MOVES[m]
        n = (a[0] + dr, a[1] + dc)
        blocked = not (0 <= n[0] < H and 0 <= n[1] < W) or grid[n[0]][n[1]] == "#"
        qs.append(noul_q(f"Is moving {m} blocked by a wall or the edge?", blocked))
    return example("grid_nav", state, qs, variant)


def graph_route(rng: random.Random, variant: int):
    n = rng.randint(5, 10)
    names = pick(rng, CITIES, n)
    edges = {}
    for i in range(1, n):  # spanning tree for connectivity
        j = rng.randrange(i)
        edges[(names[i], names[j])] = rng.randint(1, 9)
    for _ in range(rng.randint(1, n)):
        u, v = pick(rng, names, 2)
        if (u, v) not in edges and (v, u) not in edges:
            edges[(u, v)] = rng.randint(1, 9)
    adj = {x: {} for x in names}
    for (u, v), w in edges.items():
        adj[u][v] = w
        adj[v][u] = w
    src, dst = pick(rng, names, 2)
    # Dijkstra from destination gives cost-to-go for every node
    dist = {dst: 0}
    pq = [(0, dst)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e9):
            continue
        for v, w in adj[u].items():
            if d + w < dist.get(v, 1e9):
                dist[v] = d + w
                heapq.heappush(pq, (d + w, v))
    nbrs = list(adj[src])
    if len(nbrs) < 2:
        return None
    good = {v for v in nbrs if adj[src][v] + dist[v] == dist[src]}
    edge_txt = "; ".join(f"{u}-{v} ({w} km)" for (u, v), w in rng.sample(list(edges.items()), len(edges)))
    state = (f"Road network: {edge_txt}. You are in {src} and must reach {dst} with the shortest total distance."
             if variant % 2 else f"Travel planner. Current city: {src}. Destination: {dst}. Roads with lengths: {edge_txt}.")
    qs = [choice_q("Which city should you drive to next?", nbrs, [1.0 if v in good else 0.0 for v in nbrs], rng)]
    if rng.random() < 0.5:
        opts, idx = numeric_options(dist[src], rng, k=4, fmt=lambda v: f"{v} km")
        qs.append(choice_q("What is the shortest total distance?", opts, idx, rng))
    return example("graph_route", state, qs, variant)


LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _winner(b):
    for x, y, z in LINES:
        if b[x] != "." and b[x] == b[y] == b[z]:
            return b[x]
    return None


@lru_cache(maxsize=None)
def _minimax(b: str, player: str) -> int:
    w = _winner(b)
    if w:
        return 1 if w == player else -1
    if "." not in b:
        return 0
    other = "O" if player == "X" else "X"
    return max(-_minimax(b[:i] + player + b[i + 1:], other) for i, c in enumerate(b) if c == ".")


def tictactoe(rng: random.Random, variant: int):
    for _ in range(50):
        b = ["."] * 9
        turns = rng.randint(1, 6)
        p = "X"
        ok = True
        for _t in range(turns):
            empt = [i for i, c in enumerate(b) if c == "."]
            b[rng.choice(empt)] = p
            p = "O" if p == "X" else "X"
            if _winner(b):
                ok = False
                break
        if ok and b.count(".") >= 2:
            break
    else:
        return None
    bs = "".join(b)
    other = "O" if p == "X" else "X"
    empt = [i for i, c in enumerate(bs) if c == "."]
    vals = {i: -_minimax(bs[:i] + p + bs[i + 1:], other) for i in empt}
    best = max(vals.values())
    cell = lambda i: f"row {i // 3 + 1}, column {i % 3 + 1}"
    rows = "\n".join(" ".join(bs[r * 3:(r + 1) * 3]) for r in range(3))
    state = f"Tic-tac-toe board (. = empty), {p} to move:\n{rows}"
    opts = [cell(i) for i in empt]
    qs = [choice_q(f"Where should {p} play to get the best possible outcome?", opts,
                   [1.0 if vals[i] == best else 0.0 for i in empt], rng)]
    if rng.random() < 0.4:
        outcome = {1: "win", 0: "draw", -1: "loss"}[best]
        qs.append(choice_q(f"With perfect play from both sides, what is the result for {p}?", ["loss", "draw", "win"],
                           ["loss", "draw", "win"].index(outcome), rng, qtype="score"))
    return example("tictactoe", state, qs, variant)


def nim(rng: random.Random, variant: int):
    piles = [rng.randint(1, 7) for _ in range(rng.randint(2, 4))]
    moves = [(i, k) for i, h in enumerate(piles) for k in range(1, h + 1)]
    if len(moves) > 12:
        moves = rng.sample(moves, 12)
    xor = 0
    for h in piles:
        xor ^= h
    def good(m):
        i, k = m
        after = piles[:]
        after[i] -= k
        x = 0
        for h in after:
            x ^= h
        return x == 0
    g = [good(m) for m in moves]
    target = [1.0 if x else 0.0 for x in g] if any(g) else [1.0] * len(moves)  # losing position: all moves equal
    names = "ABCD"
    state = ("Nim (normal play: whoever takes the last stone wins). Piles: "
             + ", ".join(f"pile {names[i]} = {h}" for i, h in enumerate(piles)) + ". It is your move.")
    opts = [f"take {k} from pile {names[i]}" for i, k in moves]
    qs = [choice_q("Which move is winning?", opts, target, rng)]
    qs.append(noul_q("Is the player to move in a winning position?", xor != 0))
    return example("nim", state, qs, variant)


def tetris_place(rng: random.Random, variant: int):
    from verqen.environments.procedural import TetrisPlacementTask
    task = TetrisPlacementTask.sample(rng)
    if task is None:
        return None
    state, options, scores = task
    import math
    m = max(scores)
    target = [math.exp((s - m) / 1.5) for s in scores]
    return example("tetris", state, [choice_q("Which placement is best for the current piece?", options, target, rng)], variant)


FAMILIES = {
    "grid_nav": grid_nav,
    "graph_route": graph_route,
    "tictactoe": tictactoe,
    "nim": nim,
    "tetris": tetris_place,
}
