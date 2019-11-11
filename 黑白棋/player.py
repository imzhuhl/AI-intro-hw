import random
import ipdb

winner_map = {"X": 0, "O": 1}
priority_table = [[(0, 0), (0, 7), (7, 0), (7, 7)],
                  [(0, 2), (0, 5), (2, 0), (5, 0), (2, 7), (5, 7), (7, 2), (7, 5)],
                  [(2, 2), (2, 5), (5, 2), (5, 5)],
                  [(3, 0), (4, 0), (0, 3), (0, 4), (7, 3), (7, 4), (3, 7), (4, 7)],
                  [(3, 2), (4, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 3), (5, 4)],
                  [(3, 3), (4, 4), (3, 4), (4, 3)], # 0
                  [(1, 3), (1, 4), (3, 1), (4, 1), (6, 3), (6, 4), (3, 6), (4, 6)],
                  [(1, 2), (1, 5), (2, 1), (5, 1), (6, 2), (6, 5), (2, 6), (5, 6)],
                  [(0, 1), (0, 6), (7, 1), (7, 6), (1, 0), (6, 0), (1, 7), (6, 7)],
                  [(1, 1), (6, 6), (1, 6), (6, 1)]]

class Player(object):
    """
    玩家基类
    """

    def __init__(self, color=None):
        """
        获取当前玩家状态
        :param color: 如果 color=='X',则表示是黑棋一方; color=='O'，则表示白棋一方。
        """
        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘获取最佳落子位置坐标
        :param board: 当前棋盘
        :return: 落子位置坐标
        """
        pass

    def move(self, board, action):
        """
        落下棋子，根子落下的棋子坐标获取反转棋子的坐标列表
        :param board: 棋盘
        :param action: 落下棋子的坐标
        :return: 反转棋子的坐标列表
        """
        flipped_pos = board._move(action, self.color)
        return flipped_pos


class RandomPlayer(Player):
    """
    随机玩家, 随机返回一个合法落子位置
    """

    def __init__(self, color):
        """
        继承基类玩家，玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        super().__init__(color)

    def random_choice(self, board):
        """
        从合法落子位置中随机选一个落子位置
        :param board: 棋盘
        :return: 随机合法落子位置, e.g. 'A1'
        """
        # 用 list() 方法获取所有合法落子位置坐标列表
        action_list = list(board.get_legal_actions(self.color))

        # 如果 action_list 为空，则返回 None,否则从中选取一个随机元素，即合法的落子坐标
        if len(action_list) == 0:
            return None
        else:
            return random.choice(action_list)

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，{}-{} 正在思考中...".format(player_name, self.color))
        action = self.random_choice(board)
        return action


class HumanPlayer(Player):
    """
    人类玩家
    """

    def __init__(self, color):
        """
        继承基类，玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        super().__init__(color)

    def get_move(self, board):
        """
        根据当前棋盘输入人类合法落子位置
        :param board: 棋盘
        :return: 人类下棋落子位置
        """
        # 如果 self.color 是黑棋 "X",则 player 是 "黑棋"，否则是 "白棋"
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人类玩家输入落子位置，如果输入 'Q', 则返回 'Q'并结束比赛。
        # 如果人类玩家输入棋盘位置，e.g. 'A1'，
        # 首先判断输入是否正确，然后再判断是否符合黑白棋规则的落子位置
        while True:
            action = input(
                    "请'{}-{}'方输入一个合法的坐标(e.g. 'D3'，若不想进行，请务必输入'Q'结束游戏。): ".format(player,
                                                                                 self.color))

            # 如果人类玩家输入 Q 则表示想结束比赛
            if action == "Q" or action == 'q':
                return "Q"
            else:
                row, col = action[1].upper(), action[0].upper()

                # 检查人类输入是否正确
                if row in '12345678' and col in 'ABCDEFGH':
                    # 检查人类输入是否为符合规则的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的输入不合法，请重新输入!")


class AIPlayer(Player):
    """
    AI 玩家
    """

    def __init__(self, color, strategy):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        :param strategy: 0表示minimax，1表示alpha-beta剪枝
        """

        super().__init__(color)
        self.strategy = strategy

    def alpha_beta(self, board, color, alpha=-100, beta=100, depth=1):
        ac = None
        action_list = list(board.get_legal_actions(color))
        if depth == 7 or len(action_list) == 0:
            winner, diff = board.get_winner()
            if winner_map[self.color] == winner:
                return diff, None
            else:
                return -diff, None

        for action in action_list:
            flipped_pos = board._move(action, color)
            next_color = "O" if color == "X" else "X"  # 交换玩家
            
            if color == self.color:  # 我方
                val, tmp = self.alpha_beta(board, next_color, alpha=alpha, depth=depth+1)
                board.backpropagation(action, flipped_pos, color)
                if alpha < val:
                    alpha = val
                    ac = action
                if alpha >= beta:
                    return alpha, ac

            else:  # 敌方
                val, tmp = self.alpha_beta(board, next_color, beta=beta, depth=depth+1)
                board.backpropagation(action, flipped_pos, color)
                if beta > val:
                    beta = val
                    ac = action
                if alpha >= beta:
                    return beta, None

        if color == self.color:
            return alpha, ac
        else:
            return beta, ac



    def minimax(self, board, color, depth=1):
        ac = None
        action_list = list(board.get_legal_actions(color))
        if depth == 5:
            winner, diff = board.get_winner()
            if winner_map[self.color] == winner:
                return diff, None
            else:
                return -diff, None
        if color == self.color:  # 我方
            rst = -100
            for action in action_list:
                flipped_pos = board._move(action, color)

                # 交换玩家
                next_color = "O" if color == "X" else "X"
                val, tmp = self.minimax(board, next_color, depth+1)
                board.backpropagation(action, flipped_pos, color)
                if rst < val:
                    rst = val
                    ac = action
                
            return rst, ac
        else:  # 敌方
            rst = 100
            for action in action_list:
                flipped_pos = board._move(action, color)

                # 交换玩家
                next_color = "O" if color == "X" else "X"
                val, tmp = self.minimax(board, next_color, depth+1)
                board.backpropagation(action, flipped_pos, color)
                if rst > val:
                    rst = val
                    ac = action
            return rst, ac


    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会, {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        # 用 list() 方法获取所有合法落子位置坐标列表
        if self.strategy == 0:
            val, action = self.minimax(board, self.color, 1)
        if self.strategy == 1:
            val, action = self.alpha_beta(board, self.color, -100, 100, 1)

        # print(val, action)
        # board.display()
        # ------------------------------------------------------------------------
        print(action)
        return action