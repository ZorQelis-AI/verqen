"""Deterministic Standard Tetris Environment for VerQen-T1.

Implements standard 10x20 Tetris with:
- Standard 7 Tetromino shapes: I, O, T, S, Z, J, L
- Full rotation matrices (0°, 90°, 180°, 270°)
- Placement-level candidate generation (column, rotation)
- Fast bitmask/grid collision detection and line clearing
- Dellacherie heuristic feature extraction (holes, aggregate height, bumpiness)
- Structured state serialization for VerQen-T1 decision queries
- Beautiful Unicode/ANSI terminal board renderer
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from verqen.environments.base import Environment


# Tetromino piece definitions (relative coordinates at rotation 0)
TETROMINOES = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # 0 deg
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # 90 deg
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # 0 deg
    ],
    "T": [
        [(0, 0), (0, 1), (0, 2), (1, 1)],  # 0 deg
        [(0, 1), (1, 0), (1, 1), (2, 1)],  # 90 deg
        [(1, 0), (1, 1), (1, 2), (0, 1)],  # 180 deg
        [(0, 0), (1, 0), (2, 0), (1, 1)],  # 270 deg
    ],
    "L": [
        [(0, 0), (1, 0), (2, 0), (2, 1)],  # 0 deg
        [(0, 0), (0, 1), (0, 2), (1, 0)],  # 90 deg
        [(0, 0), (0, 1), (1, 1), (2, 1)],  # 180 deg
        [(1, 0), (1, 1), (1, 2), (0, 2)],  # 270 deg
    ],
    "J": [
        [(0, 1), (1, 1), (2, 1), (2, 0)],  # 0 deg
        [(0, 0), (1, 0), (1, 1), (1, 2)],  # 90 deg
        [(0, 0), (0, 1), (1, 0), (2, 0)],  # 180 deg
        [(0, 0), (0, 1), (0, 2), (1, 2)],  # 270 deg
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],  # 0 deg
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # 90 deg
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],  # 0 deg
        [(0, 1), (1, 0), (1, 1), (2, 0)],  # 90 deg
    ],
}


class TetrisEnv(Environment):
    """Standard 10x20 Tetris Environment designed for VerQen-T1 inference."""

    WIDTH = 10
    HEIGHT = 20

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = [[0] * self.WIDTH for _ in range(self.HEIGHT)]
        self.current_piece: str = "T"
        self.next_piece: str = "I"
        self.score: int = 0
        self.lines_cleared: int = 0
        self.pieces_dropped: int = 0
        self.done: bool = False
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> str:
        """Reset board to empty and spawn initial pieces."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.board = [[0] * self.WIDTH for _ in range(self.HEIGHT)]
        self.current_piece = self._random_piece()
        self.next_piece = self._random_piece()
        self.score = 0
        self.lines_cleared = 0
        self.pieces_dropped = 0
        self.done = False
        return self.observe()

    def _random_piece(self) -> str:
        return self.rng.choice(list(TETROMINOES.keys()))

    def get_column_heights(self, board: Optional[List[List[int]]] = None) -> List[int]:
        """Compute the height of each of the 10 columns."""
        b = board if board is not None else self.board
        heights = [0] * self.WIDTH
        for c in range(self.WIDTH):
            for r in range(self.HEIGHT):
                if b[r][c] != 0:
                    heights[c] = self.HEIGHT - r
                    break
        return heights

    def get_holes(self, board: Optional[List[List[int]]] = None) -> int:
        """Count empty cells covered by at least one filled cell above."""
        b = board if board is not None else self.board
        holes = 0
        for c in range(self.WIDTH):
            found_block = False
            for r in range(self.HEIGHT):
                if b[r][c] != 0:
                    found_block = True
                elif found_block and b[r][c] == 0:
                    holes += 1
        return holes

    def get_bumpiness(self, board: Optional[List[List[int]]] = None) -> int:
        """Compute sum of absolute height differences between adjacent columns."""
        heights = self.get_column_heights(board)
        return sum(abs(heights[i] - heights[i + 1]) for i in range(self.WIDTH - 1))

    def get_valid_placements(self, piece: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compute all valid (column, rotation) landing positions for the piece."""
        p_name = piece or self.current_piece
        rotations = TETROMINOES[p_name]
        valid = []

        for rot_idx, shape in enumerate(rotations):
            p_width = max(c for r, c in shape) + 1
            for col in range(self.WIDTH - p_width + 1):
                landing_row = self._find_landing_row(shape, col)
                if landing_row is not None:
                    # Check if landing causes game over (exceeds top)
                    if landing_row < 0:
                        continue
                    # Simulate landing to compute heuristic value
                    sim_b, lines_c = self._simulate_placement(shape, col, landing_row)
                    val = self._evaluate_board(sim_b, landing_row, lines_c)
                    action_str = f"Col {col}, Rot {rot_idx * 90}"
                    valid.append({
                        "name": action_str,
                        "col": col,
                        "rot_idx": rot_idx,
                        "landing_row": landing_row,
                        "heuristic_score": val,
                        "lines_cleared": lines_c,
                    })

        # Sort by heuristic score descending
        valid.sort(key=lambda x: x["heuristic_score"], reverse=True)
        return valid

    def _find_landing_row(self, shape: List[Tuple[int, int]], col: int) -> Optional[int]:
        """Find the lowest row where the shape can land without collision."""
        # Start from top and drop down
        for r in range(-3, self.HEIGHT):
            # Check collision at row r + 1
            if self._check_collision(shape, r + 1, col):
                return r
        return self.HEIGHT - 1

    def _check_collision(self, shape: List[Tuple[int, int]], row: int, col: int) -> bool:
        """Return True if shape collides with walls, floor, or filled cells."""
        for r_offset, c_offset in shape:
            br = row + r_offset
            bc = col + c_offset
            if bc < 0 or bc >= self.WIDTH:
                return True
            if br >= self.HEIGHT:
                return True
            if br >= 0 and self.board[br][bc] != 0:
                return True
        return False

    def _simulate_placement(
        self, shape: List[Tuple[int, int]], col: int, row: int
    ) -> Tuple[List[List[int]], int]:
        """Simulate placing piece and clearing lines on a copy of the board."""
        sim = [row_cells[:] for row_cells in self.board]
        for r_off, c_off in shape:
            br = row + r_off
            bc = col + c_off
            if 0 <= br < self.HEIGHT and 0 <= bc < self.WIDTH:
                sim[br][bc] = 1

        # Check line clears
        new_b = [r for r in sim if any(cell == 0 for cell in r)]
        lines_cleared = self.HEIGHT - len(new_b)
        while len(new_b) < self.HEIGHT:
            new_b.insert(0, [0] * self.WIDTH)
        return new_b, lines_cleared

    def _evaluate_board(self, board: List[List[int]], landing_height: int, lines_cleared: int) -> float:
        """Pierre Dellacherie standard heuristic for AI Tetris evaluation."""
        heights = self.get_column_heights(board)
        agg_height = sum(heights)
        holes = self.get_holes(board)
        bumpiness = self.get_bumpiness(board)

        # Standard proven weights:
        # +3.4188 * lines_cleared - 0.510066 * agg_height - 0.35663 * holes - 0.184483 * bumpiness
        score = (
            3.5 * lines_cleared
            - 0.51 * agg_height
            - 0.85 * holes
            - 0.20 * bumpiness
            - 0.10 * (self.HEIGHT - landing_height)
        )
        return round(score, 4)

    def observe(self) -> str:
        """Format current game state into a structured VerQen state prompt."""
        heights = self.get_column_heights()
        holes = self.get_holes()
        bumpiness = self.get_bumpiness()
        max_h = max(heights)

        obs = (
            f"Tetris 10x20 Board State | Score: {self.score:,} | Lines: {self.lines_cleared} | "
            f"Pieces Dropped: {self.pieces_dropped} | Max Column Height: {max_h}/20 | "
            f"Column Heights: {heights} | Active Holes: {holes} | Surface Bumpiness: {bumpiness} | "
            f"Current Falling Piece: [{self.current_piece}-shape] | Next Preview: [{self.next_piece}-shape]"
        )
        return obs

    def valid_actions(self) -> List[str]:
        placements = self.get_valid_placements()
        return [p["name"] for p in placements]

    def step(self, action_name: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Apply the chosen placement action to the board."""
        if self.done:
            return self.observe(), 0.0, True, {"reason": "game_over"}

        placements = {p["name"]: p for p in self.get_valid_placements()}
        if action_name not in placements:
            # Fallback to the top placement if invalid choice was made
            fallback = self.get_valid_placements()[0]
            p_data = fallback
            reward = -5.0
        else:
            p_data = placements[action_name]
            reward = 1.0

        shape = TETROMINOES[self.current_piece][p_data["rot_idx"]]
        col = p_data["col"]
        row = p_data["landing_row"]

        # Check if placement overflows top of board
        if row < 0 or any((row + r_off) < 0 for r_off, _ in shape):
            self.done = True
            return self.observe(), -10.0, True, {"reason": "board_overflow"}

        # Place piece on actual board
        for r_off, c_off in shape:
            br = row + r_off
            bc = col + c_off
            if 0 <= br < self.HEIGHT and 0 <= bc < self.WIDTH:
                self.board[br][bc] = 1

        # Clear lines
        new_b = [r for r in self.board if any(c == 0 for c in r)]
        cleared = self.HEIGHT - len(new_b)
        while len(new_b) < self.HEIGHT:
            new_b.insert(0, [0] * self.WIDTH)
        self.board = new_b

        # Update stats
        self.pieces_dropped += 1
        self.lines_cleared += cleared
        line_points = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_points.get(cleared, cleared * 200) + 10
        reward += cleared * 10.0

        # Cycle pieces
        self.current_piece = self.next_piece
        self.next_piece = self._random_piece()

        # Check game over (no valid moves remaining for new piece)
        if not self.get_valid_placements():
            self.done = True

        info = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_dropped": self.pieces_dropped,
            "lines_this_turn": cleared,
        }
        return self.observe(), reward, self.done, info

    def is_terminal(self) -> bool:
        """Return True if episode has terminated."""
        return self.done

    def reward(self) -> float:
        """Return cumulative score reward."""
        return float(self.score)

    def render_ascii(self) -> str:
        """Render a clean cross-platform ASCII representation of the board."""
        lines = []
        lines.append("  +" + "--" * self.WIDTH + "+")
        for r in range(self.HEIGHT):
            row_str = "  |"
            for c in range(self.WIDTH):
                if self.board[r][c] != 0:
                    row_str += "[]"
                else:
                    row_str += " ."
            row_str += "|"
            lines.append(row_str)
        lines.append("  +" + "--" * self.WIDTH + "+")
        lines.append(f"  Score: {self.score:,} | Lines: {self.lines_cleared} | Piece: [{self.current_piece}] -> Next: [{self.next_piece}]")
        return "\n".join(lines)
