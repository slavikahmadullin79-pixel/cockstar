"""Проверки на реальном скриншоте chess.com и полная партия в симуляторе.

Запуск: python -m pytest tests  (нужен stockfish в PATH или рядом с bot.py)
"""
import os
import random
import sys

import chess
import chess.engine
import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from bot import ChessBot, PROMOTION_ORDER, find_engine  # noqa: E402
from vision import (BOARD_COLORS_RGB, HIGHLIGHT_COLORS_RGB, BoardReader,  # noqa: E402
                    board_grid, find_board)

SCREENSHOT = os.path.join(HERE, 'samples', 'start.png')


def load():
    return cv2.imread(SCREENSHOT)


def test_reads_start_position():
    img = load()
    rect = find_board(img)
    assert rect is not None
    reader = BoardReader(rect)
    assert reader.calibrate(img)
    assert reader.white_bottom
    assert reader.read(img) == board_grid(chess.Board())


class FakeChessCom:
    """Рисует доску в стиле chess.com и принимает клики, как сайт."""

    def __init__(self, bot_color, seed=0, drop_first_click=False):
        self.base = load()
        self.rect = find_board(self.base)
        reader = BoardReader(self.rect)
        assert reader.calibrate(self.base)
        self.sq = reader.sq
        self.sprites = {}
        for r in (0, 1, 6, 7):
            for c in range(8):
                sym = chess.Board().piece_at(reader.square_at(r, c)).symbol()
                cell = reader.cell(self.base, r, c).copy()
                bg = cell[2, 2].astype(np.int16)
                mask = np.abs(cell.astype(np.int16) - bg).sum(axis=2) > 45
                self.sprites.setdefault(sym, (cell, mask))
        self.white_bottom = bot_color == chess.WHITE
        self.board = chess.Board()
        self.rng = random.Random(seed)
        self.selected = None
        self.awaiting_promotion = None
        self.drop_next = drop_first_click
        self.bot_color = bot_color
        if bot_color == chess.BLACK:
            self.opponent_move()

    def opponent_move(self):
        if not self.board.is_game_over():
            self.board.push(self.rng.choice(list(self.board.legal_moves)))

    def cell_rc(self, square):
        f, rank = chess.square_file(square), chess.square_rank(square)
        return (7 - rank, f) if self.white_bottom else (rank, 7 - f)

    def grab(self):
        img = self.base.copy()
        x, y, _ = self.rect
        last = self.board.move_stack[-1] if self.board.move_stack else None
        for sq in chess.SQUARES:
            r, c = self.cell_rc(sq)
            x0, y0 = round(x + c * self.sq), round(y + r * self.sq)
            x1, y1 = round(x + (c + 1) * self.sq), round(y + (r + 1) * self.sq)
            light = (r + c) % 2 == 0
            colors = HIGHLIGHT_COLORS_RGB if last and sq in (last.from_square, last.to_square) else BOARD_COLORS_RGB
            img[y0:y1, x0:x1] = colors[0 if light else 1][::-1]
            piece = self.board.piece_at(sq)
            if piece:
                sprite, mask = self.sprites[piece.symbol()]
                h, w = min(y1 - y0, sprite.shape[0]), min(x1 - x0, sprite.shape[1])
                region = img[y0:y0 + h, x0:x0 + w]
                region[mask[:h, :w]] = sprite[:h, :w][mask[:h, :w]]
        return img

    def click(self, px, py):
        if self.drop_next:
            self.drop_next = False
            return
        x, y, _ = self.rect
        c, r = int((px - x) // self.sq), int((py - y) // self.sq)
        if self.awaiting_promotion:
            move = self.awaiting_promotion
            to_r, _ = self.cell_rc(move.to_square)
            move.promotion = PROMOTION_ORDER[r - to_r]
            self.awaiting_promotion = None
            self.board.push(move)
            self.opponent_move()
            return
        sq = chess.square(c, 7 - r) if self.white_bottom else chess.square(7 - c, r)
        if self.selected is None:
            self.selected = sq
            return
        move, self.selected = chess.Move(self.selected, sq), None
        piece = self.board.piece_at(move.from_square)
        if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
            self.awaiting_promotion = move
            return
        assert move in self.board.legal_moves, f'бот сделал нелегальный ход {move}'
        self.board.push(move)
        self.opponent_move()


@pytest.fixture(scope='module')
def engine():
    path = find_engine(None)
    if not path:
        pytest.skip('stockfish не найден')
    eng = chess.engine.SimpleEngine.popen_uci(path)
    yield eng
    eng.quit()


@pytest.mark.parametrize('color', [chess.WHITE, chess.BLACK])
def test_full_game_in_simulator(engine, color):
    site = FakeChessCom(color, seed=int(color), drop_first_click=True)
    bot = ChessBot(site.grab, site.click, engine, chess.engine.Limit(time=0.02),
                   poll=0, retry_after=0, sleep=lambda s: None)
    final = bot.play_game()
    assert final.move_stack == site.board.move_stack
    assert final.is_checkmate() and final.turn != color, final.result()


def test_promotion_click_picks_requested_piece():
    site = FakeChessCom(chess.WHITE)
    site.board = chess.Board('8/P6k/8/8/8/8/8/K7 w - - 0 1')
    reader = BoardReader(site.rect)
    reader.white_bottom = True
    ChessBot(site.grab, site.click, None, None, sleep=lambda s: None).make_move(
        reader, chess.Move.from_uci('a7a8n'))
    assert site.board.move_stack[0] == chess.Move.from_uci('a7a8n')
