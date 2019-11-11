# 导入玩家基类
from player import Player, RandomPlayer, HumanPlayer, AIPlayer
from board import Board
from game import Game
import random



if __name__ == "__main__":
    # black_player = RandomPlayer("X")
    # white_player = AIPlayer("O", 1)
    black_player = AIPlayer("X", 0)
    white_player = AIPlayer("O", 1)


    # 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
    game = Game(black_player, white_player)
    # 开始下棋
    winner_id, result, diff = game.run()
