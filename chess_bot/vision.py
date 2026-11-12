"""Распознавание доски chess.com на скриншоте.

Фигуры не зашиты в код: шаблоны снимаются с экрана в начальной позиции,
поэтому распознавание подстраивается под ваш набор фигур и масштаб.
"""
import chess
import cv2
import numpy as np

# Цвета темы «Зелёная» на chess.com (RGB): светлая клетка, тёмная клетка,
# подсветка последнего хода на светлой и тёмной клетке.
BOARD_COLORS_RGB = [(235, 236, 208), (115, 149, 82)]
HIGHLIGHT_COLORS_RGB = [(245, 246, 130), (185, 202, 67)]

EMPTY_FRACTION = 0.06  # доля «не-фоновых» пикселей, ниже которой клетка пустая
FEATURE_SIZE = 32


def _bgr(colors):
    return [np.array(c[::-1], dtype=np.int16) for c in colors]


def _color_mask(img, colors, tol):
    img16 = img.astype(np.int16)
    mask = np.zeros(img.shape[:2], dtype=bool)
    for c in _bgr(colors):
        mask |= np.abs(img16 - c).sum(axis=2) < tol
    return mask


def find_board(img, board_colors=None, tol=18):
    """Ищет доску по цветам клеток. Возвращает (x, y, size) или None."""
    colors = (board_colors or BOARD_COLORS_RGB) + HIGHLIGHT_COLORS_RGB
    mask = _color_mask(img, colors, tol).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 160 or h < 160 or not 0.92 < w / h < 1.08:
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    return x, y, min(w, h)


def analyze_cell(cell, board_colors=None):
    """Возвращает (доля пикселей фигуры, вектор-признак, средняя яркость фигуры)."""
    h, w = cell.shape[:2]
    b = max(2, h // 20)
    ring = np.concatenate([
        cell[:b].reshape(-1, 3), cell[-b:].reshape(-1, 3),
        cell[:, :b].reshape(-1, 3), cell[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(ring, axis=0).astype(np.int16)
    c16 = cell.astype(np.int16)
    fg = np.abs(c16 - bg).sum(axis=2) > 45
    # Буквы и цифры координат нарисованы цветом соседней клетки — это не фигура.
    fg &= ~_color_mask(cell, board_colors or BOARD_COLORS_RGB, 30)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    frac = float(fg.mean())
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    feat = np.where(fg, gray, 128.0).astype(np.float32)
    feat = cv2.resize(feat, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    brightness = float(gray[fg].mean()) if fg.any() else 0.0
    return frac, feat, brightness


class BoardReader:
    """Связывает область доски на экране с клетками шахматной доски."""

    def __init__(self, rect, board_colors=None):
        self.x, self.y, self.size = rect
        self.sq = self.size / 8
        self.board_colors = board_colors
        self.white_bottom = True
        self.templates = []  # [(символ фигуры, признак)]

    # --- геометрия -------------------------------------------------------
    def cell(self, img, r, c):
        x0, y0 = round(self.x + c * self.sq), round(self.y + r * self.sq)
        x1, y1 = round(self.x + (c + 1) * self.sq), round(self.y + (r + 1) * self.sq)
        return img[y0:y1, x0:x1]

    def square_at(self, r, c):
        """Ряд/колонка на экране (0 — сверху/слева) -> клетка chess."""
        if self.white_bottom:
            return chess.square(c, 7 - r)
        return chess.square(7 - c, r)

    def screen_pos(self, square):
        """Центр клетки в координатах скриншота."""
        f, rank = chess.square_file(square), chess.square_rank(square)
        c, r = (f, 7 - rank) if self.white_bottom else (7 - f, rank)
        return self.x + (c + 0.5) * self.sq, self.y + (r + 0.5) * self.sq

    # --- распознавание ---------------------------------------------------
    def _analyze_all(self, img):
        return {(r, c): analyze_cell(self.cell(img, r, c), self.board_colors)
                for r in range(8) for c in range(8)}

    def calibrate(self, img):
        """Снимает шаблоны фигур, считая, что на экране (почти) начальная позиция.

        Допускается, что соперник белыми уже сделал первый ход: опустевшие
        клетки просто пропускаются. Возвращает True, если удалось.
        """
        cells = self._analyze_all(img)
        occupied = {k for k, v in cells.items() if v[0] >= EMPTY_FRACTION}
        if any((r, c) not in occupied for r in (0, 7) for c in (0, 2, 3, 4, 5, 7)):
            return False
        middle = sum(1 for r in range(2, 6) for c in range(8) if (r, c) in occupied)
        if middle > 2:
            return False

        top = np.mean([cells[(0, c)][2] for c in range(8) if (0, c) in occupied])
        bottom = np.mean([cells[(7, c)][2] for c in range(8) if (7, c) in occupied])
        self.white_bottom = bottom > top

        start = chess.Board()
        templates = []
        for r in (0, 1, 6, 7):
            for c in range(8):
                if (r, c) in occupied:
                    sym = start.piece_at(self.square_at(r, c)).symbol()
                    templates.append((sym, cells[(r, c)][1]))
        if len({s for s, _ in templates}) != 12:
            return False
        self.templates = templates
        return True

    def read(self, img):
        """Кортеж из 64 символов по порядку chess.SQUARES ('.' — пусто)."""
        grid = ['.'] * 64
        for (r, c), (frac, feat, _) in self._analyze_all(img).items():
            if frac < EMPTY_FRACTION:
                continue
            best = min(self.templates, key=lambda t: float(np.abs(t[1] - feat).mean()))
            grid[self.square_at(r, c)] = best[0]
        return tuple(grid)


def board_grid(board):
    """Тот же формат, что у BoardReader.read, но из объекта chess.Board."""
    return tuple(p.symbol() if (p := board.piece_at(sq)) else '.' for sq in chess.SQUARES)
