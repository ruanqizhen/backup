import torch
from torch.autograd import Variable

import math

from ChessBoard import ChessBoard


class NNPlayer:
    @staticmethod
    def chessboard_feature(board, chess_color):
        return [[0.999 if j == chess_color else 0.001 for j in i] for i in board]

    @staticmethod
    def move_feature(move):
        move_board = [[0.001 for i in range(ChessBoard.SIZE)] for j in range(ChessBoard.SIZE)]
        move_board[move[0]][move[1]] = 0.999
        return move_board

    def __init__(self, color, model=None):
        self.color = color
        self.model = model if model else torch.load("model.bin")

    def choose_move(self, board, valid_positions):

        my_chess_board = NNPlayer.chessboard_feature(board, self.color)
        opponent_color = ChessBoard.get_opponent(self.color)
        opponent_chess_board = NNPlayer.chessboard_feature(board, opponent_color)

        score = -math.inf
        result = None
        info = []
        for move in valid_positions:
            move_board = NNPlayer.move_feature(move)

            feature = Variable(torch.tensor(
                [i for j in my_chess_board for i in j]
                + [i for j in opponent_chess_board for i in j]
                + [i for j in move_board for i in j]
            ).type(torch.FloatTensor), requires_grad=False)

            pred = self.model(feature)
            info.append((move, pred.item()))

            if pred.item() > score:
                result = move
                score = pred.item()

        print(info)
        # print(board)
        return result
