"""Шахматный бот: смотрит на доску chess.com, думает Stockfish'ем и ходит мышью.

Запуск: python bot.py [--time 0.5] [--elo 2000] [--engine путь\\к\\stockfish.exe]
Остановка: Ctrl+C в консоли или резко увести мышь в левый верхний угол экрана.
"""
import argparse
import glob
import os
import shutil
import sys
import time

import chess
import chess.engine

from vision import BoardReader, board_grid, find_board

# chess.com показывает выбор фигуры столбиком от поля превращения вниз.
PROMOTION_ORDER = [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


class ChessBot:
    def __init__(self, grab, click, engine, limit, poll=0.15, retry_after=3.0, sleep=time.sleep):
        self.grab = grab      # () -> BGR-скриншот
        self.click = click    # (x, y) в координатах скриншота
        self.engine = engine
        self.limit = limit
        self.poll = poll
        self.retry_after = retry_after
        self.sleep = sleep

    # --- чтение экрана ---------------------------------------------------
    def stable_read(self, reader, attempts=10):
        """Два одинаковых кадра подряд — значит, анимация хода закончилась."""
        prev = reader.read(self.grab())
        for _ in range(attempts):
            self.sleep(min(self.poll, 0.1))
            cur = reader.read(self.grab())
            if cur == prev:
                return cur
            prev = cur
        return None

    def wait_for_start(self):
        """Ждёт начала партии. Возвращает (reader, board, наш цвет)."""
        log('Жду начальную позицию на доске...')
        while True:
            img = self.grab()
            rect = find_board(img)
            if rect:
                reader = BoardReader(rect)
                if reader.calibrate(img):
                    grid = self.stable_read(reader)
                    board = chess.Board()
                    if grid == board_grid(board):
                        pass
                    else:
                        # Соперник белыми мог успеть сделать первый ход.
                        move = self.find_move(board, grid)
                        if move is None:
                            self.sleep(0.5)
                            continue
                        board.push(move)
                    me = chess.WHITE if reader.white_bottom else chess.BLACK
                    log(f'Доска найдена {rect}, играю {"белыми" if me else "чёрными"}')
                    return reader, board, me
            self.sleep(0.5)

    @staticmethod
    def find_move(board, grid):
        if grid is None:
            return None
        for move in board.legal_moves:
            board.push(move)
            ok = board_grid(board) == grid
            board.pop()
            if ok:
                return move
        return None

    # --- действия --------------------------------------------------------
    def make_move(self, reader, move):
        self.click(*reader.screen_pos(move.from_square))
        self.sleep(0.05)
        self.click(*reader.screen_pos(move.to_square))
        if move.promotion:
            self.sleep(0.35)
            x, y = reader.screen_pos(move.to_square)
            self.click(x, y + PROMOTION_ORDER.index(move.promotion) * reader.sq)

    def play_game(self):
        reader, board, me = self.wait_for_start()
        pending = None  # время нашего последнего хода, пока он не виден на экране
        last_warn = 0.0

        while not board.is_game_over():
            if board.turn == me and pending is None:
                move = self.engine.play(board, self.limit).move
                log(f'Мой ход: {board.san(move)}')
                self.make_move(reader, move)
                board.push(move)
                pending = time.monotonic()
                continue

            grid = self.stable_read(reader)
            if grid == board_grid(board):
                pending = None
                self.sleep(self.poll)
                continue

            move = self.find_move(board, grid)
            if move is not None:
                log(f'Ход соперника: {board.san(move)}')
                board.push(move)
                pending = None
                continue

            if pending is not None:
                ours = board.pop()
                if grid == board_grid(board):
                    if time.monotonic() - pending > self.retry_after:
                        log('Ход не прошёл, пробую ещё раз')
                        pending = None
                        continue
                    # Ход ещё не отрисовался — ждём.
                    board.push(ours)
                    self.sleep(self.poll)
                    continue
                board.push(ours)

            if time.monotonic() - last_warn > 3:
                diff = [chess.square_name(sq) for sq in chess.SQUARES
                        if grid and grid[sq] != board_grid(board)[sq]]
                log(f'Не понимаю позицию на экране (отличия: {diff or "кадр нестабилен"}), жду...')
                last_warn = time.monotonic()
            self.sleep(self.poll)

        log(f'Партия окончена: {board.result()}')
        return board


def find_engine(path):
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    found = glob.glob(os.path.join(here, '**', 'stockfish*.exe'), recursive=True)
    return found[0] if found else shutil.which('stockfish')


def main():
    ap = argparse.ArgumentParser(description='Бот для партий против ботов chess.com')
    ap.add_argument('--engine', help='путь к stockfish.exe (по умолчанию ищется рядом с bot.py)')
    ap.add_argument('--time', type=float, default=0.5, help='секунд на ход (по умолчанию 0.5)')
    ap.add_argument('--elo', type=int, help='ограничить силу Stockfish (1320–3190)')
    ap.add_argument('--monitor', type=int, default=1, help='номер монитора (1 — основной)')
    ap.add_argument('--games', type=int, default=0, help='сколько партий сыграть (0 — без ограничения)')
    args = ap.parse_args()

    if sys.platform == 'win32':
        import ctypes
        try:  # реальные пиксели при масштабе Windows 125%/150%
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()

    import mss
    import numpy as np
    import pyautogui

    pyautogui.PAUSE = 0.02

    engine_path = find_engine(args.engine)
    if not engine_path:
        sys.exit('Не найден Stockfish: положите stockfish.exe в папку с ботом или укажите --engine')
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    if args.elo:
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': args.elo})

    sct = mss.mss()
    mon = sct.monitors[args.monitor]

    def grab():
        return np.array(sct.grab(mon))[:, :, :3]

    def click(x, y):
        pyautogui.click(mon['left'] + round(x), mon['top'] + round(y))

    bot = ChessBot(grab, click, engine, chess.engine.Limit(time=args.time))
    played = 0
    try:
        while not args.games or played < args.games:
            bot.play_game()
            played += 1
            log('Жду следующую партию (Ctrl+C — выход)')
            time.sleep(3)
    except (KeyboardInterrupt, pyautogui.FailSafeException):
        log('Остановлено')
    finally:
        engine.quit()


if __name__ == '__main__':
    main()
