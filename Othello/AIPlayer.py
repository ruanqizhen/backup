from b78player import *
from ChessBoard import ChessBoard


class AIPlayer:
    def __init__(self, color, model=None):
        self.color = color
        self.player = Player('W') if color == ChessBoard.CHESS_WHITE else Player('B')

    def choose_move(self, board, valid_positions=None):
        player_board = [
            ['B' if a == ChessBoard.CHESS_BLACK else 'W' if a == ChessBoard.CHESS_WHITE else ChessBoard.CHESS_EMPTY
             for a in b] for b in board]
        return self.player.chooseMove(player_board)
