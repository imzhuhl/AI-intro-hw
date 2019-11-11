# -*- coding: utf-8 -*-

# from func_timeout import func_timeout, FunctionTimedOut
import datetime
from board import Board


class Game(object):
    def __init__(self, black_player, white_player):
        self.board = Board()  # 棋盘
        # 定义棋盘上当前下棋棋手，先默认是 None
        self.current_player = None
        self.black_player = black_player  # 黑棋一方
        self.white_player = white_player  # 白棋一方
        self.black_player.color = "X"
        self.white_player.color = "O"

    def switch_player(self, black_player, white_player):
        """
        游戏过程中切换玩家
        :param black_player: 黑棋
        :param white_player: 白棋
        :return: 当前玩家
        """
        # 如果当前玩家是 None 或者 白棋一方 white_player，则返回 黑棋一方 black_player;
        if self.current_player is None:
            return black_player
        else:
            # 如果当前玩家是黑棋一方 black_player 则返回 白棋一方 white_player
            if self.current_player == self.black_player:
                return white_player
            else:
                return black_player

    def print_winner(self, winner):
        """
        打印赢家
        :param winner: [0,1,2] 分别代表黑棋获胜、白棋获胜、平局3种可能。
        :return:
        """
        print(['黑棋获胜!', '白棋获胜!', '平局'][winner])

    def force_loss(self, is_timeout=True):
        """
         落子3个不合符规则和超时则结束游戏
        :param is_timeout: 判断是否是时间超时,默认是时间超时
        :return:
        """

        if self.current_player == self.black_player:
            win_color = '白棋 - O'
            loss_color = '黑棋 - X'
            winner = 1
        else:
            win_color = '黑棋 - X'
            loss_color = '白棋 - O'
            winner = 0
        if is_timeout:
            print('\n{} 思考超过 60s, {} 胜'.format(loss_color, win_color))
        else:
            print('\n{} 落子 3 次不符合规则,故 {} 胜'.format(loss_color, win_color))
        diff = 0

        return winner, diff

    def run(self):
        """
        运行游戏
        :return:
        """
        # 定义统计双方下棋时间
        total_time = {"X": 0, "O": 0}
        # 定义双方每一步下棋时间
        step_time = {"X": 0, "O": 0}
        # 初始化胜负结果和棋子差
        winner = None
        diff = -1

        # 游戏开始
        print('\n=====开始游戏!=====\n')
        # 棋盘初始化
        self.board.display(step_time, total_time)
        while True:
            # 切换当前玩家,如果当前玩家是 None 或者白棋 white_player，则返回黑棋 black_player;
            #  否则返回 white_player。
            self.current_player = self.switch_player(self.black_player, self.white_player)
            start_time = datetime.datetime.now()
            # 当前玩家对棋盘进行思考后，得到落子位置
            for i in range(0, 3):
                # 判断当前下棋方
                color = "X" if self.current_player == self.black_player else "O"
                # 获取当前下棋方合法落子位置
                legal_actions = list(self.board.get_legal_actions(color))
                # 获取落子位置
                action = self.current_player.get_move(self.board)

                # 如果 action 是 Q 则说明人类想结束比赛
                if action == "Q" or action is None:
                    # 说明人类想结束游戏，即根据棋子个数定输赢。
                    # winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                    break

                # 判断当前下棋方落子是否符合合法落子,如果不合法,则需要对方重新输入
                if action not in legal_actions:
                    print("你落子不符合规则,请重新落子！")
                    continue
                else:
                    # 落子合法则直接 break
                    break
            else:
                # 落子3次不合法，结束游戏！
                winner, diff = self.force_loss(is_timeout=False)
                break
            end_time = datetime.datetime.now()
            if action == "Q":
                # 说明人类想结束游戏，即根据棋子个数定输赢。
                winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                break
            if action is None:
                continue
            else:
                if len(action) == 0:
                    continue
                # 当前玩家执行招法，改变棋盘
                color = "X" if self.current_player == self.black_player else "O"
                self.board._move(action, color)

            # 统计一步所用的时间
            es_time = (end_time - start_time).seconds
            # 统计每种棋子下棋所用总时间
            if self.current_player == self.black_player:
                # 当前选手是黑棋一方
                step_time["X"] = es_time
                total_time["X"] += es_time
            else:
                step_time["O"] = es_time
                total_time["O"] += es_time
            # 显示当前棋盘
            self.board.display(step_time, total_time)

            # 根据当前棋盘，判断棋局是否终止
            # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
            b_list = list(self.board.get_legal_actions('X'))
            w_list = list(self.board.get_legal_actions('O'))

            is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

            # 判断游戏是否结束
            if is_over:
                winner, diff = self.board.get_winner()  # 得到赢家 0,1,2
                break

        print('\n=====游戏结束!=====\n')
        self.board.display(step_time, total_time)
        self.print_winner(winner)

        # 返回'black_win','white_win','draw',棋子数差
        if winner is not None and diff > -1:
            result = {0: 'black_win', 1: 'white_win', 2: 'draw'}[winner]

            return winner, result, diff


# if __name__ == '__main__':
#     from Human_player import HumanPlayer
#     from Random_player import RandomPlayer

#     x = HumanPlayer("X")
#     o = RandomPlayer("O")
#     game = Game(x, o)
#     game.run()
