class ChessBoard:
    SIZE = 8
    CHESS_EMPTY = 0
    CHESS_WHITE = 1
    CHESS_BLACK = 2

    # 重置棋盘
    @staticmethod
    def initial_board():
        board = [[ChessBoard.CHESS_EMPTY for i in range(ChessBoard.SIZE)] for j in range(ChessBoard.SIZE)]
        board[3][3] = ChessBoard.CHESS_BLACK
        board[3][4] = ChessBoard.CHESS_WHITE
        board[4][3] = ChessBoard.CHESS_WHITE
        board[4][4] = ChessBoard.CHESS_BLACK
        return board

    # 是否出界
    @staticmethod
    def is_position_inside_board(position):
        x, y = position
        return 0 <= x < ChessBoard.SIZE and 0 <= y < ChessBoard.SIZE

    @staticmethod
    def get_opponent(chess):
        return ChessBoard.CHESS_BLACK if chess == ChessBoard.CHESS_WHITE else ChessBoard.CHESS_WHITE if chess == ChessBoard.CHESS_BLACK else chess

    # 是否是合法走法
    @staticmethod
    def is_valid_position(board, chess, position):
        x_target, y_target = position
        # 如果该位置已经有棋子或者出界了，返回False
        if not ChessBoard.is_position_inside_board(position):
            return False

        if board[x_target][y_target] != ChessBoard.CHESS_EMPTY:
            return False

        opponent = ChessBoard.get_opponent(chess)

        for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            x = x_target + x_direction
            y = y_target + y_direction
            if not ChessBoard.is_position_inside_board((x, y)):
                continue
            if board[x][y] != opponent:
                continue

            x += x_direction
            y += y_direction
            while ChessBoard.is_position_inside_board((x, y)) and board[x][y] == opponent:
                x += x_direction
                y += y_direction

            if ChessBoard.is_position_inside_board((x, y)) and board[x][y] == chess:
                return True

        return False

    # 获取可落子的位置
    @staticmethod
    def get_all_valid_positions(board, chess):
        positions = []
        for x in range(ChessBoard.SIZE):
            for y in range(ChessBoard.SIZE):
                if ChessBoard.is_valid_position(board, chess, (x, y)):
                    positions.append((x, y))
        return positions

    # 获取棋盘上黑白双方的棋子数
    @staticmethod
    def get_score(board):
        b = 0
        w = 0
        for x in range(ChessBoard.SIZE):
            for y in range(ChessBoard.SIZE):
                if board[x][y] == ChessBoard.CHESS_BLACK:
                    b += 1
                if board[x][y] == ChessBoard.CHESS_WHITE:
                    w += 1
        return {ChessBoard.CHESS_BLACK: b, ChessBoard.CHESS_WHITE: w}

    # 放一个棋子
    @staticmethod
    def put(board, chess, position):
        x_target, y_target = position
        if ChessBoard.is_valid_position(board, chess, position):
            board[x_target][y_target] = chess
            opponent = ChessBoard.get_opponent(chess)
            for x_direction, y_direction in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
                x = x_target + x_direction
                y = y_target + y_direction

                while ChessBoard.is_position_inside_board((x, y)) and board[x][y] == opponent:
                    x += x_direction
                    y += y_direction

                if ChessBoard.is_position_inside_board((x, y)) and board[x][y] == chess:
                    x = x_target + x_direction
                    y = y_target + y_direction

                    while ChessBoard.is_position_inside_board((x, y)) and board[x][y] == opponent:
                        board[x][y] = chess
                        x += x_direction
                        y += y_direction
        return board
