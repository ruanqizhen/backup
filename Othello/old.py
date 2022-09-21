async def log_one_line(f, board, prediction):
    log = ""
    for x in range(ChessBoard.SIZE):
        for y in range(ChessBoard.SIZE):
            log += f"{board[x][y]}"
        log += " "
    for x in range(ChessBoard.SIZE):
        for y in range(ChessBoard.SIZE):
            log += f"{prediction[x][y]}"
        log += " "
    log += "\n"
    await f.write(log)


async def log_move(f, board, move, candidates, color):
    if color == ChessBoard.CHESS_WHITE:
        board = [[ChessBoard.get_opponent(i) for i in j] for j in board]
    prediction = [[0 for i in range(ChessBoard.SIZE)] for j in range(ChessBoard.SIZE)]
    for pos in candidates:
        prediction[pos[0]][pos[1]] = 1
    prediction[move[0]][move[1]] = 2

    for i in range(4):
        await log_one_line(f, board, prediction)
        board = rotate(board)
        prediction = rotate(prediction)

    board = board[::-1]
    prediction = prediction[::-1]

    for i in range(4):
        await log_one_line(f, board, prediction)
        board = rotate(board)
        prediction = rotate(prediction)


async def collect_ai_data():
    # app = Reversi()
    # app.pack()
    # app.mainloop()

    player1 = AIPlayer(ChessBoard.CHESS_BLACK)
    player2 = AIPlayer(ChessBoard.CHESS_WHITE)
    async with aiofiles.open('log.txt', 'a') as f:
        for q in range(1000):
            print(f"Round {q}")
            game_over = False
            board = ChessBoard.initial_board()
            current_player = player1
            step = 0
            while not game_over:
                step += 1
                candidates = ChessBoard.get_all_valid_positions(board, current_player.color)
                if step < 30 and random.random() < 0.3:
                    # random player
                    move = random.choice(candidates)
                else:
                    move = current_player.choose_move(board)
                    if step > 25:
                        await log_move(f, board, move, candidates, current_player.color)
                board = ChessBoard.put(board, current_player.color, move[0], move[1])
                opponent_color = ChessBoard.get_opponent(current_player.color)
                if ChessBoard.get_all_valid_positions(board, opponent_color):
                    # switch player
                    current_player = player1 if current_player == player2 else player2
                    # print(f"Step: {step}, Current player: {current_player.color}")
                else:
                    if not ChessBoard.get_all_valid_positions(board, current_player.color):
                        # print("Game Over!")
                        print(ChessBoard.get_score(board))
                        game_over = True
                        break


async def dedup_data():
    seen = set()
    async with aiofiles.open('log.txt', 'r') as fin, aiofiles.open('log_unique.txt', 'w') as fout:
        async for line in fin:
            h = hash(line)
            if h not in seen:
                await fout.write(line)
                seen.add(h)


async def dump_one_weight(f, w):
    await f.write("\n".join([" ".join([str(i.item()) for i in j]) for j in w]))
    await f.write("\n")


async def dump_weights(w1, w2, w3):
    async with aiofiles.open('weight.txt', 'w') as f:
        await dump_one_weight(f, w1)
        await dump_one_weight(f, w2)
        await dump_one_weight(f, w3)


async def load_weights():
    async with aiofiles.open('weight.txt', 'r') as f:
        w = [[float(i) for i in line.split()] for line in await f.readlines()]
        w1 = Variable(torch.tensor(w[:64]).type(torch.FloatTensor), requires_grad=True)
        w2 = Variable(torch.tensor(w[64:128]).type(torch.FloatTensor), requires_grad=True)
        w3 = Variable(torch.tensor(w[128:]).type(torch.FloatTensor), requires_grad=True)
        return w1, w2, w3


async def training_set():
    i = 0
    x = []
    y = []
    async with aiofiles.open('training.txt', 'r') as f:
        async for line in f:
            line = "".join(line.split())
            if len(line) == 128:
                feature = [int(c) + 1.0 for c in line[:64]]
                count_negative_value = line[64:].count('1')
                label = [1.0 if c == '0' else 10.0 if c == '2' else -10.0 / count_negative_value for c in line[64:]]
                i += 1
                x.append(feature)
                y.append(label)
                if i > 32:
                    i = 0
                    x_tensor = Variable(torch.tensor(x).type(torch.FloatTensor), requires_grad=False)
                    y_tensor = Variable(torch.tensor(y).type(torch.FloatTensor), requires_grad=False)
                    x = []
                    y = []
                    yield x_tensor, y_tensor


def random_weights():
    d_in, h, d_out = 64, 64, 64
    w1 = Variable(torch.randn(d_in, h).type(torch.FloatTensor), requires_grad=True)
    w2 = Variable(torch.randn(h, h).type(torch.FloatTensor), requires_grad=True)
    w3 = Variable(torch.randn(h, d_out).type(torch.FloatTensor), requires_grad=True)
    return w1, w2, w3


async def test_training_set():
    w1, w2, w3 = await load_weights()
    print(w1)
    print(w2)
    print(w3)


async def train_model_manual():
    w1, w2, w3 = await load_weights()
    # w1, w2, w3 = random_weights()
    learning_rate = 0.000001

    i = 0
    loss = -1.0
    async for x, y in training_set():
        i += 1
        if i % 1000 == 2:
            await dump_weights(w1, w2, w3)
            print(i, loss.item())
            # print(y_pred)
        for t in range(100):
            # 正向传递:使用变量上的运算来计算预测的y; 这些
            # 与我们用于计算使用张量的正向传递完全相同,
            # 但我们不需要保留对中间值的引用,
            # 因为我们没有实现向后传递.
            y_pred = x.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)

            # 使用变量上的操作计算和打印损失.
            # 现在损失是形状变量 (1,) 并且 loss.data 是形状的张量
            # (1,); loss.data[0] 是持有损失的标量值.
            loss = (y_pred - y).pow(2).sum()
            # if t % 100 == 0:
            #     # print(y_pred.data.numpy().flatten().tolist())
            #     print(t, loss.data)
            # 使用autograd来计算反向传递.
            # 该调用将使用requires_grad = True来计算相对于所有变量的损失梯度.
            # 在这次调用之后 w1.grad 和 w2.grad 将是变量
            # 它们分别相对于w1和w2保存损失的梯度.
            loss.backward()

            # 使用梯度下降更新权重; w1.data 和 w2.data 是张量,
            # w1.grad 和 w2.grad 是变量并且 w1.grad.data 和 w2.grad.data
            # 是张量.
            w1.data -= learning_rate * w1.grad.data * 0.0005
            w2.data -= learning_rate * w2.grad.data * 0.02
            w3.data -= learning_rate * w3.grad.data

            # 更新权重后手动将梯度归零
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            w3.grad.data.zero_()
        # print(y_pred.data.numpy().flatten().tolist())
        # break


async def train_model():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    d_in, h, d_out = 64, 10, 64
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, d_out),
    )
    # model = torch.load("model.bin")
    # model.to(device)

    loss_fn = torch.nn.MSELoss(size_average=False)
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters())

    i = 0
    loss = -1.0

    print("loading data...")
    data = []
    async for x, y in training_set():
        data.append((x, y))
        break
    print("data loaded.")

    for s in range(20000):
        for (x, y) in data:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # torch.save(model, "model.bin")
        if s % 100 == 0:
            print(s, loss.item())


async def train_model_cnn():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = torch.nn.Sequential(
    #     torch.nn.Conv2d(1, 8, kernel_size=5), torch.nn.ReLU(),
    #     torch.nn.Conv2d(8, 64, kernel_size=3), torch.nn.ReLU(),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(64*4, 64), torch.nn.ReLU(),
    #     torch.nn.Linear(64, 64),
    # )
    model = torch.load("model.bin")
    # model.to(device)

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adadelta(model.parameters())

    print("loading data...")
    data = []
    async for x, y in training_set():
        data.append((x.view(-1, 1, 8, 8), y))
        break

    print("data loaded.")

    for s in range(1000):
        for (x, y) in data:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model, "model.bin")
        if s % 1 == 0:
            print(s, loss.item())