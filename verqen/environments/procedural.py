"""Randomized multi-step decision environments for RLCD.

Every environment exposes the same text interface a deployed VerQen agent sees:

    env.reset(rng)
    obs = env.observe()          -> {"state": str, "question": str, "options": [str, ...]}
    reward, done = env.step(i)   -> choose option index i
    env.oracle()                 -> indices of optimal options (or None if unknown)

Every episode is procedurally generated (random mazes, graphs, sites, boards), so a
policy cannot memorize an action script; the old fixed browser/tool envs could be.
"""

from __future__ import annotations

import heapq
import random
from typing import Any

from verqen.data.procedural.agents import ITEM_KIND, SECTIONS, VERBS
from verqen.data.procedural.common import CITIES
from verqen.data.procedural.games import MOVES, _bfs, _minimax, _winner
from verqen.environments.games.tetris import TetrisEnv


class DecisionEnv:
    name = "base"
    max_steps = 20

    def reset(self, rng: random.Random) -> None: ...
    def observe(self) -> dict[str, Any]: ...
    def step(self, i: int) -> tuple[float, bool]: ...
    def oracle(self) -> list[int] | None: return None


class GridNavEnv(DecisionEnv):
    name = "grid_nav"

    def reset(self, rng):
        self.rng = rng
        while True:
            H, W = rng.randint(5, 9), rng.randint(5, 9)
            self.grid = [["#" if rng.random() < 0.22 else "." for _ in range(W)] for _ in range(H)]
            cells = [(r, c) for r in range(H) for c in range(W) if self.grid[r][c] == "."]
            if len(cells) < 6:
                continue
            self.pos, self.goal = rng.sample(cells, 2)
            self.dist = _bfs(self.grid, self.pos, self.goal)
            if self.pos in self.dist and self.dist[self.pos] >= 3:
                break
        self.H, self.W = H, W
        self.t = 0
        self.max_steps = 2 * self.dist[self.pos] + 4
        self.opts = list(MOVES)

    def observe(self):
        g = [row[:] for row in self.grid]
        g[self.goal[0]][self.goal[1]] = "G"
        g[self.pos[0]][self.pos[1]] = "A"
        board = "\n".join(" ".join(r) for r in g)
        return {"state": f"Grid map ({self.H}x{self.W}). A = agent, G = goal, # = wall. Row 0 is the top.\n{board}",
                "question": "What is the best next move?", "options": self.opts}

    def step(self, i):
        self.t += 1
        dr, dc = MOVES[self.opts[i]]
        n = (self.pos[0] + dr, self.pos[1] + dc)
        if n in self.dist:
            reward = 0.1 if self.dist[n] < self.dist[self.pos] else -0.1
            self.pos = n
        else:
            reward = -0.2
        if self.pos == self.goal:
            return reward + 1.0, True
        return reward, self.t >= self.max_steps

    def oracle(self):
        out = []
        for i, m in enumerate(self.opts):
            dr, dc = MOVES[m]
            n = (self.pos[0] + dr, self.pos[1] + dc)
            if n in self.dist and self.dist[n] == self.dist[self.pos] - 1:
                out.append(i)
        return out


class GraphRouteEnv(DecisionEnv):
    name = "graph_route"

    def reset(self, rng):
        n = rng.randint(6, 10)
        self.names = rng.sample(CITIES, n)
        self.adj = {x: {} for x in self.names}
        for i in range(1, n):
            j = rng.randrange(i)
            w = rng.randint(1, 9)
            self.adj[self.names[i]][self.names[j]] = w
            self.adj[self.names[j]][self.names[i]] = w
        for _ in range(n):
            u, v = rng.sample(self.names, 2)
            if v not in self.adj[u]:
                w = rng.randint(1, 9)
                self.adj[u][v] = w
                self.adj[v][u] = w
        self.cur, self.dst = rng.sample(self.names, 2)
        self.cost = {self.dst: 0}
        pq = [(0, self.dst)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > self.cost.get(u, 1e9):
                continue
            for v, w in self.adj[u].items():
                if d + w < self.cost.get(v, 1e9):
                    self.cost[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        self.t = 0
        self.max_steps = 3 * n

    def observe(self):
        roads = "; ".join(f"{u}-{v} ({w} km)" for u in self.names for v, w in self.adj[u].items() if u < v)
        self.opts = sorted(self.adj[self.cur])
        return {"state": f"Road network: {roads}. You are in {self.cur}; destination {self.dst}.",
                "question": "Which city should you drive to next?", "options": self.opts}

    def step(self, i):
        self.t += 1
        v = self.opts[i]
        reward = -self.adj[self.cur][v] / 10.0
        self.cur = v
        if v == self.dst:
            return reward + 2.0, True
        return reward, self.t >= self.max_steps

    def oracle(self):
        return [i for i, v in enumerate(self.opts) if self.adj[self.cur][v] + self.cost[v] == self.cost[self.cur]]


class BrowserEnv(DecisionEnv):
    """Randomized web task: optional login and cookie banner, then home -> section -> item -> action."""
    name = "browser"

    def reset(self, rng):
        self.rng = rng
        self.sections = rng.sample(SECTIONS, rng.randint(4, 7))
        self.sec = rng.choice(self.sections)
        kind = ITEM_KIND[self.sec]
        self.items = [f"{kind} #{rng.randint(1000, 9999)}" for _ in range(rng.randint(3, 6))]
        self.item = rng.choice(self.items)
        self.verbs = rng.sample(VERBS, rng.randint(3, 5))
        self.verb = rng.choice(self.verbs)
        self.page = "login" if rng.random() < 0.3 else "home"
        self.banner = rng.random() < 0.3
        self.typed_user = False
        self.here = None
        self.t = 0
        self.max_steps = 10

    def _actions(self):
        if self.banner:
            return ["click button 'Accept all cookies'", "click button 'Reject non-essential cookies'"]
        if self.page == "login":
            return ["type into input 'Username'", "type into input 'Password'", "click button 'Sign in'"]
        if self.page == "home":
            return [f"click link '{s}'" for s in self.sections]
        if self.page == "section":
            items = self.items if self.here == self.sec else self.other_items
            return [f"click link '{i}'" for i in items] + ["click button 'Back'"]
        if self.page == "item":
            return [f"click button '{v}'" for v in self.verbs] + ["click button 'Back'"]
        return ["click button 'Back'", "click link 'Home'"]

    def observe(self):
        self.opts = self._actions()
        where = {"login": "/login", "home": "/home", "section": f"/{(self.here or self.sec).lower()}",
                 "item": f"item page of {self.here}", "other": f"/{(self.here or 'x').lower()}"}[self.page]
        extra = " A cookie banner covers the page." if self.banner else ""
        extra += " Username already entered." if self.page == "login" and self.typed_user else ""
        return {"state": f"Browser agent. Task: {self.verb} {self.item} from {self.sec}. Policy: decline optional cookies. "
                         f"Current page: {where}.{extra}",
                "question": "Which action should the agent take next?", "options": self.opts}

    def _correct(self) -> set[str]:
        if self.banner:
            return {"click button 'Reject non-essential cookies'"}
        if self.page == "login":
            return {"click button 'Sign in'"} if self.typed_user else {"type into input 'Username'"}
        if self.page == "home":
            return {f"click link '{self.sec}'"}
        if self.page == "section":
            return {f"click link '{self.item}'"} if self.here == self.sec else {"click button 'Back'"}
        if self.page == "item":
            return {f"click button '{self.verb}'"} if self.here == self.item else {"click button 'Back'"}
        return {"click button 'Back'", "click link 'Home'"}

    def step(self, i):
        self.t += 1
        a = self.opts[i]
        good = a in self._correct()
        reward = 0.2 if good else -0.2
        if self.banner:
            self.banner = False
            if "Accept" in a:
                reward = -0.5  # privacy violation
        elif self.page == "login":
            if "Username" in a:
                self.typed_user = True
            elif "Sign in" in a and self.typed_user:
                self.page = "home"
        elif self.page == "home":
            self.page, self.here = "section", a.split("'")[1]
            kind = ITEM_KIND[self.here]
            self.other_items = [f"{kind} #{self.rng.randint(1000, 9999)}" for _ in range(3)]
        elif self.page == "section":
            if "Back" in a:
                self.page, self.here = "home", None
            else:
                self.page, self.here = "item", a.split("'")[1]
        elif self.page == "item":
            if "Back" in a:
                self.page, self.here = "section", self.sec
            elif self.here == self.item and a == f"click button '{self.verb}'":
                return reward + 1.0, True
            else:
                reward = -0.5  # destructive wrong action
        return reward, self.t >= self.max_steps

    def oracle(self):
        good = self._correct()
        return [i for i, a in enumerate(self.opts) if a in good]


class TicTacToeEnv(DecisionEnv):
    name = "tictactoe"

    def reset(self, rng):
        self.rng = rng
        self.b = ["."] * 9
        self.me = rng.choice("XO")
        self.opp = "O" if self.me == "X" else "X"
        if self.me == "O":
            self._opp_move()
        self.max_steps = 5

    def _opp_move(self):
        empt = [i for i, c in enumerate(self.b) if c == "."]
        if not empt:
            return
        if self.rng.random() < 0.5:  # half-optimal opponent
            s = "".join(self.b)
            best = max(empt, key=lambda i: -_minimax(s[:i] + self.opp + s[i + 1:], self.me))
            self.b[best] = self.opp
        else:
            self.b[self.rng.choice(empt)] = self.opp

    def observe(self):
        self.empt = [i for i, c in enumerate(self.b) if c == "."]
        s = "".join(self.b)
        rows = "\n".join(" ".join(s[r * 3:(r + 1) * 3]) for r in range(3))
        return {"state": f"Tic-tac-toe board (. = empty), {self.me} to move:\n{rows}",
                "question": f"Where should {self.me} play?", "options": [f"row {i // 3 + 1}, column {i % 3 + 1}" for i in self.empt]}

    def step(self, i):
        self.b[self.empt[i]] = self.me
        if _winner(self.b) == self.me:
            return 1.0, True
        if "." not in self.b:
            return 0.5, True
        self._opp_move()
        if _winner(self.b) == self.opp:
            return 0.0, True
        if "." not in self.b:
            return 0.5, True
        return 0.0, False

    def oracle(self):
        s = "".join(self.b)
        vals = [-_minimax(s[:i] + self.me + s[i + 1:], self.opp) for i in self.empt]
        best = max(vals)
        return [k for k, v in enumerate(vals) if v == best]


def _placement_text(p: dict, env: TetrisEnv) -> str:
    return f"column {p['col']}, rotation {p['rot_idx'] * 90}: clears {p['lines_cleared']} line(s), lands at row {p['landing_row']}"


class TetrisPlacementTask:
    @staticmethod
    def board_after_play(rng: random.Random) -> TetrisEnv:
        env = TetrisEnv(seed=rng.randint(0, 10**9))
        for _ in range(rng.randint(0, 25)):
            pl = env.get_valid_placements()
            if not pl or env.done:
                break
            env.step(rng.choice(pl[: rng.randint(1, 4)])["name"])  # imperfect play gives realistic boards
        return env

    @staticmethod
    def describe(env: TetrisEnv, placements: list[dict]) -> tuple[str, list[str]]:
        state = env.render_ascii() + f"\nCurrent piece: {env.current_piece}. Column heights: {env.get_column_heights()}. Holes: {env.get_holes()}."
        opts = []
        for p in placements:
            shape = env._simulate_placement(TETROMINOES_ROT(env.current_piece, p["rot_idx"]), p["col"], p["landing_row"])[0]
            opts.append(f"{_placement_text(p, env)}, leaves {env.get_holes(shape)} hole(s), max height {max(env.get_column_heights(shape))}")
        return state, opts

    @classmethod
    def sample(cls, rng: random.Random):
        env = cls.board_after_play(rng)
        if env.done:
            return None
        pl = env.get_valid_placements()
        if len(pl) < 3:
            return None
        chosen = [pl[0]] + rng.sample(pl[1:], min(len(pl) - 1, rng.randint(2, 5)))
        state, opts = cls.describe(env, chosen)
        return state, opts, [p["heuristic_score"] for p in chosen]


def TETROMINOES_ROT(piece: str, rot_idx: int):
    from verqen.environments.games.tetris import TETROMINOES
    return TETROMINOES[piece][rot_idx]


class TetrisPlayEnv(DecisionEnv):
    name = "tetris"

    def reset(self, rng):
        self.rng = rng
        self.env = TetrisPlacementTask.board_after_play(rng)
        self.t = 0
        self.max_steps = 15

    def observe(self):
        pl = self.env.get_valid_placements()
        self.pl = [pl[0]] + self.rng.sample(pl[1:], min(len(pl) - 1, 5)) if pl else []
        self.rng.shuffle(self.pl)
        state, opts = TetrisPlacementTask.describe(self.env, self.pl)
        return {"state": state, "question": "Which placement is best for the current piece?", "options": opts}

    def step(self, i):
        self.t += 1
        before = self.env.lines_cleared
        _, _, done, _ = self.env.step(self.pl[i]["name"])
        reward = (self.env.lines_cleared - before) * 1.0 + 0.05 - (1.0 if done else 0.0)
        done = done or not self.env.get_valid_placements()
        return reward, done or self.t >= self.max_steps

    def oracle(self):
        best = max(p["heuristic_score"] for p in self.pl)
        return [i for i, p in enumerate(self.pl) if p["heuristic_score"] == best]


ENVIRONMENTS = {cls.name: cls for cls in (GridNavEnv, GraphRouteEnv, BrowserEnv, TicTacToeEnv, TetrisPlayEnv)}
