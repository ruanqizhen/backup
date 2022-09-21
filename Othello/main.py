import tkinter as tk
import random
from ChessBoard import ChessBoard
from AIPlayer import AIPlayer
from NNPlayer import NNPlayer
import torch
from torch.autograd import Variable
from copy import deepcopy
from json import JSONEncoder
import json
from torch.utils.data import Dataset


class ChessBoardCanvas(tk.Canvas):
    cell_size = 46
    margin = 5
    board = ChessBoard.initial_board()
    current = ChessBoard.CHESS_BLACK
    # ai = AIPlayer(ChessBoard.CHESS_WHITE)
    ai = NNPlayer(ChessBoard.CHESS_WHITE)

    def __init__(self, master):
        width = ChessBoard.SIZE * self.cell_size
        tk.Canvas.__init__(self, master, relief=tk.RAISED, bd=4, bg='white', width=width, height=width, cursor="cross")
        self.bind("<Button-1>", self.put_stone)

        for i in range(ChessBoard.SIZE):
            for j in range(ChessBoard.SIZE):
                background_color = "#5FA758"
                y0 = i * self.cell_size + self.margin
                x0 = j * self.cell_size + self.margin
                self.create_rectangle(x0, y0, x0 + self.cell_size, y0 + self.cell_size, fill=background_color, width=1)

        self.refresh()

    def put_stone(self, event):  # 放置棋子
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        # 获得坐标
        j = int(x / self.cell_size)
        i = int(y / self.cell_size)
        if ChessBoard.is_valid_position(self.board, self.current, (i, j)):
            print(i, j)
            self.board = ChessBoard.put(self.board, self.current, (i, j))
            opponent = ChessBoard.get_opponent(self.current)
            if ChessBoard.get_all_valid_positions(self.board, opponent):
                self.current = ChessBoard.get_opponent(self.current)
                print(f"Current player: {self.current}")
            else:
                if not ChessBoard.get_all_valid_positions(self.board, self.current):
                    print("Game Over!")
                    print(ChessBoard.get_score(self.board))
            self.refresh()

        if self.current == ChessBoard.CHESS_WHITE:
            self.after(100, self.ai_move)

    def ai_move(self):
        while self.current == ChessBoard.CHESS_WHITE:
            valid_positions = ChessBoard.get_all_valid_positions(self.board, self.current)
            print(self.board)
            print(valid_positions)
            move = self.ai.choose_move(self.board, valid_positions)
            self.board = ChessBoard.put(self.board, self.current, move)
            opponent = ChessBoard.get_opponent(self.current)
            if ChessBoard.get_all_valid_positions(self.board, opponent):
                self.current = ChessBoard.get_opponent(self.current)
                print(f"Current player: {self.current}")
            else:
                if not ChessBoard.get_all_valid_positions(self.board, self.current):
                    print("Game Over!")
                    print(ChessBoard.get_score(self.board))
                    self.refresh()
                    break
            self.refresh()

    def refresh(self):
        for i in range(ChessBoard.SIZE):
            for j in range(ChessBoard.SIZE):
                y0 = i * self.cell_size + self.margin
                x0 = j * self.cell_size + self.margin

                if self.board[i][j] == ChessBoard.CHESS_BLACK:
                    chess_color = "#000000"
                elif self.board[i][j] == ChessBoard.CHESS_WHITE:
                    chess_color = "#D0D0D0"
                else:
                    continue
                self.create_oval(x0 + 2, y0 + 2, x0 + self.cell_size - 2, y0 + self.cell_size - 2, fill=chess_color,
                                 width=0)


class Reversi(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master.title("黑白棋")
        self.f_board = ChessBoardCanvas(self)
        self.f_board.pack(padx=10, pady=10)


def rotate(board):
    return list(zip(*board[::-1]))


def ui():
    app = Reversi()
    app.pack()
    app.mainloop()


def train_one_step(model, loss_fn, optimizer, board, move, chess_color, label, step):
    my_chess_board = NNPlayer.chessboard_feature(board, chess_color)
    opponent_color = ChessBoard.get_opponent(chess_color)
    opponent_chess_board = NNPlayer.chessboard_feature(board, opponent_color)
    move_board = NNPlayer.move_feature(move)

    x = []

    for t in range(4):
        my_chess_board = rotate(my_chess_board)
        opponent_chess_board = rotate(opponent_chess_board)
        move_board = rotate(move_board)
        x.append(
            [i for j in my_chess_board for i in j]
            + [i for j in opponent_chess_board for i in j]
            + [i for j in move_board for i in j]
        )

    my_chess_board = my_chess_board[::-1]
    opponent_chess_board = opponent_chess_board[::-1]
    move_board = move_board[::-1]

    for t in range(4):
        my_chess_board = rotate(my_chess_board)
        opponent_chess_board = rotate(opponent_chess_board)
        move_board = rotate(move_board)
        x.append(
            [i for j in my_chess_board for i in j]
            + [i for j in opponent_chess_board for i in j]
            + [i for j in move_board for i in j]
        )
    y = [[0.99 if label else -0.99] for i in range(8)]

    # x.append(
    #     [i for j in my_chess_board for i in j]
    #     + [i for j in opponent_chess_board for i in j]
    #     + [i for j in move_board for i in j]
    # )
    # y = [[1 if label else 0] for i in range(1)]
    x = Variable(torch.tensor(x).type(torch.FloatTensor), requires_grad=False)
    y = Variable(torch.tensor(y).type(torch.FloatTensor), requires_grad=False)

    for s in range(2):  # step // 5):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(step, loss.item())


def train_one_step_for_ai(model, loss_fn, optimizer, board, move_neg, move_pos, chess_color):
    my_chess_board = NNPlayer.chessboard_feature(board, chess_color)
    opponent_color = ChessBoard.get_opponent(chess_color)
    opponent_chess_board = NNPlayer.chessboard_feature(board, opponent_color)
    move_neg_board = NNPlayer.move_feature(move_neg)
    move_pos_board = NNPlayer.move_feature(move_pos)

    x = []
    for t in range(4):
        my_chess_board = rotate(my_chess_board)
        opponent_chess_board = rotate(opponent_chess_board)
        move_neg_board = rotate(move_neg_board)
        move_pos_board = rotate(move_pos_board)
        x.append(
            [i for j in my_chess_board for i in j]
            + [i for j in opponent_chess_board for i in j]
            + [i for j in move_neg_board for i in j]
        )
        x.append(
            [i for j in my_chess_board for i in j]
            + [i for j in opponent_chess_board for i in j]
            + [i for j in move_pos_board for i in j]
        )

    my_chess_board = my_chess_board[::-1]
    opponent_chess_board = opponent_chess_board[::-1]
    move_neg_board = move_neg_board[::-1]
    move_pos_board = move_pos_board[::-1]

    for t in range(4):
        my_chess_board = rotate(my_chess_board)
        opponent_chess_board = rotate(opponent_chess_board)
        move_neg_board = rotate(move_neg_board)
        move_pos_board = rotate(move_pos_board)
        x.append(
            [i for j in my_chess_board for i in j]
            + [i for j in opponent_chess_board for i in j]
            + [i for j in move_neg_board for i in j]
        )
        x.append(
            [i for j in my_chess_board for i in j]
            + [i for j in opponent_chess_board for i in j]
            + [i for j in move_pos_board for i in j]
        )

    x = Variable(torch.tensor(x).type(torch.FloatTensor), requires_grad=False)

    y = [[-0.99], [0.99], [-0.99], [0.99], [-0.99], [0.99], [-0.99], [0.99], [-0.99], [0.99], [-0.99],
         [0.99], [-0.99], [0.99], [-0.99], [0.99]]
    y = Variable(torch.tensor(y).type(torch.FloatTensor), requires_grad=False)

    for s in range(1):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(move_neg, move_pos, y_pred.data)


def train_by_only_ais():
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(64 * 3, 64), torch.nn.Tanh(),
    #     torch.nn.Linear(64, 1), torch.nn.Tanh(),
    # )
    model = torch.load("model.bin")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    players = {
        0: AIPlayer(ChessBoard.CHESS_BLACK, model),
        1: AIPlayer(ChessBoard.CHESS_WHITE, model),
    }
    for q in range(20000):
        board = ChessBoard.initial_board()
        current_player = q % 2
        step = 0
        while True:
            step += 1
            candidates = ChessBoard.get_all_valid_positions(board, players[current_player].color)
            if random.random() < 0.1:
                # random player
                move = random.choice(candidates)
            else:
                move = players[current_player].choose_move(board, candidates)
                for candidate in candidates:
                    if move != candidate:
                        train_one_step_for_ai(model, loss_fn, optimizer, board, candidate, move,
                                              players[current_player].color)

            board = ChessBoard.put(board, players[current_player].color, move)
            opponent_color = ChessBoard.get_opponent(players[current_player].color)
            if ChessBoard.get_all_valid_positions(board, opponent_color):
                # switch player
                current_player = 1 - current_player
            else:
                if not ChessBoard.get_all_valid_positions(board, players[current_player].color):
                    # print("Game Over!")
                    score = ChessBoard.get_score(board)
                    break
        if q % 10 == 9:
            print(f"Round {q}")
            print(score)
            torch.save(model, "model.bin")


def train_by_players():
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(64 * 3, 64), torch.nn.Sigmoid(),
    #     torch.nn.Linear(64, 1), torch.nn.Sigmoid(),
    # )
    model = torch.load("model.bin")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    groups = [
        {
            0: AIPlayer(ChessBoard.CHESS_BLACK, model),
            1: NNPlayer(ChessBoard.CHESS_WHITE, model),
        },
        {
            0: NNPlayer(ChessBoard.CHESS_BLACK, model),
            1: AIPlayer(ChessBoard.CHESS_WHITE, model),
        },
    ]

    for q in range(20000):
        players = groups[0] if q % 4 < 2 else groups[1]

        records = {0: [], 1: []}
        board = ChessBoard.initial_board()
        current_player = q % 2
        step = 0
        winner = None
        while True:
            step += 1
            candidates = ChessBoard.get_all_valid_positions(board, players[current_player].color)
            move = players[current_player].choose_move(board, candidates)
            if step > 10:
                records[current_player].append((deepcopy(board), deepcopy(move)))
            board = ChessBoard.put(board, players[current_player].color, move)
            opponent_color = ChessBoard.get_opponent(players[current_player].color)
            if ChessBoard.get_all_valid_positions(board, opponent_color):
                # switch player
                current_player = 1 - current_player
                # print(f"Step: {step}, Current player: {current_player.color}")
            else:
                if not ChessBoard.get_all_valid_positions(board, players[current_player].color):
                    # print("Game Over!")
                    score = ChessBoard.get_score(board)
                    if score[players[0].color] > score[players[1].color]:
                        winner = 0
                    elif score[players[0].color] < score[players[1].color]:
                        winner = 1
                    break
        if winner == 0 or winner == 1:
            for p in [0, 1]:
                for i, record in enumerate(records[p]):
                    train_one_step(model, loss_fn, optimizer, record[0], record[1], players[p].color, winner == p, i)
        # print(winner)
        # print(records[winner][20])
        # print(records[1-winner][15])
        if q % 10 == 9:
            print(f"Round {q}")
            print(score)
            torch.save(model, "model.bin")


def train_by_nnplayers():
    model = torch.load("model.bin")
    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adadelta(model.parameters())

    players = {
        0: NNPlayer(ChessBoard.CHESS_BLACK, model),
        1: NNPlayer(ChessBoard.CHESS_WHITE, model),
    }

    for q in range(2000000):
        records = {0: [], 1: []}
        board = ChessBoard.initial_board()
        current_player = 0
        step = 0
        winner = None
        while True:
            step += 1
            candidates = ChessBoard.get_all_valid_positions(board, players[current_player].color)
            if step < 40 and random.random() < 0.01:
                move = random.choice(candidates)
            else:
                move = players[current_player].choose_move(board, candidates)

            if step > 20:
                records[current_player].append((deepcopy(board), deepcopy(move)))
            board = ChessBoard.put(board, players[current_player].color, move)
            opponent_color = ChessBoard.get_opponent(players[current_player].color)
            if ChessBoard.get_all_valid_positions(board, opponent_color):
                # switch player
                current_player = 1 - current_player
                # print(f"Step: {step}, Current player: {current_player.color}")
            else:
                if not ChessBoard.get_all_valid_positions(board, players[current_player].color):
                    # print("Game Over!")
                    score = ChessBoard.get_score(board)
                    if score[players[0].color] > score[players[1].color]:
                        winner = 0
                    elif score[players[0].color] < score[players[1].color]:
                        winner = 1
                    break
        if winner == 0 or winner == 1:
            for p in [0, 1]:
                for i, record in enumerate(records[p]):
                    train_one_step(model, loss_fn, optimizer, record[0], record[1], players[p].color, winner == p, i)
        # print(winner)
        # print(records[winner][20])
        # print(records[1-winner][15])
        if q % 100 == 99:
            print(f"Round {q}")
            print(score)
            torch.save(model, "model.bin")


TEST_BOARD = [[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 2, 2, 2, 0, 0],
              [0, 0, 2, 2, 2, 2, 1, 1],
              [0, 2, 2, 2, 2, 2, 1, 1],
              [0, 2, 2, 1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2, 2, 2, 0],
              [0, 0, 1, 1, 1, 2, 0, 0],
              [2, 2, 2, 2, 2, 2, 2, 0]]


def test_trainer():
    model = torch.load("model.bin")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    board = TEST_BOARD

    # train_after_ai(model, loss_fn, optimizer, board, (1, 1), (5, 1), ChessBoard.CHESS_WHITE)

    train_one_step(model, loss_fn, optimizer, board, (6, 7), ChessBoard.CHESS_WHITE, False, 20)
    train_one_step(model, loss_fn, optimizer, board, (0, 3), ChessBoard.CHESS_WHITE, True, 20)

    torch.save(model, "model.bin")


def test_player():
    color = ChessBoard.CHESS_WHITE
    ai = NNPlayer(color)
    valid_positions = ChessBoard.get_all_valid_positions(TEST_BOARD, color)
    print(valid_positions)
    move = ai.choose_move(TEST_BOARD, valid_positions)
    print(move)


def compare_2_models(players):
    board = ChessBoard.initial_board()
    current_player = 0
    while True:
        candidates = ChessBoard.get_all_valid_positions(board, players[current_player].color)
        move = players[current_player].choose_move(board, candidates)
        board = ChessBoard.put(board, players[current_player].color, move)
        opponent_color = ChessBoard.get_opponent(players[current_player].color)
        if ChessBoard.get_all_valid_positions(board, opponent_color):
            # switch player
            current_player = 1 - current_player
            # print(f"Step: {step}, Current player: {current_player.color}")
        else:
            if not ChessBoard.get_all_valid_positions(board, players[current_player].color):
                # print("Game Over!")
                score = ChessBoard.get_score(board)
                if score[players[0].color] > score[players[1].color]:
                    return 0
                elif score[players[0].color] < score[players[1].color]:
                    return 1
                return -1


def tournament():
    score = {i: 0 for i in range(10)}
    b_players = [NNPlayer(ChessBoard.CHESS_BLACK, torch.load(f"{i}.bin")) for i in range(10)]
    w_players = [NNPlayer(ChessBoard.CHESS_WHITE, torch.load(f"{i}.bin")) for i in range(10)]

    for i in range(10):
        for j in range(10):
            if i != j:
                winner = compare_2_models([b_players[i], w_players[j]])
                if winner == 0:
                    score[i] += 1
                if winner == 1:
                    score[j] += 1

    return score


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(JSONEncoder, self).default(obj)


if __name__ == '__main__':
    # train_by_players()
    # train_by_nnplayers()
    # ui()
    # test_trainer()
    # test_player()

    # train_by_only_ais()
    # print(tournament())

    # model = torch.load("model.bin")
    # # with open('weights.json', 'w') as json_file:
    # #     json.dump(model.state_dict(), json_file, cls=EncodeTensor)
    # feature = feature = Variable(torch.tensor([0.002 for i in range(192)]).type(torch.FloatTensor), requires_grad=False)
    # print(model(feature))

    def string_concat(a: str, b: str) -> str:
        return a + b

    print(string_concat(2, 3))
