#!/usr/bin/Anaconda3/python
# -*- coding: utf-8 -*-


class Board(object):
    """
    Board 黑白棋棋盘，规格是8*8，黑棋用 X 表示，白棋用 O 表示，未落子时用 . 表示。
    """

    def __init__(self):
        """
        初始化棋盘状态
        """
        self.empty = '.'  # 未落子状态
        self._board = [[self.empty for _ in range(8)] for _ in range(8)]  # 规格：8*8
        self._board[3][4] = 'X'  # 黑棋棋子
        self._board[4][3] = 'X'  # 黑棋棋子
        self._board[3][3], self._board[4][4] = 'O', 'O'  # 白棋棋子

    def __getitem__(self, index):
        """
        添加Board[][] 索引语法
        :param index: 下标索引
        :return:
        """
        return self._board[index]

    def display(self, step_time=None, total_time=None):
        """
        打印棋盘
        :param step_time: 每一步的耗时, 比如:{"X":1,"O":0},默认值是None
        :param total_time: 总耗时, 比如:{"X":1,"O":0},默认值是None
        :return:
        """
        board = self._board
        # print(step_time,total_time)
        # 打印列名
        print(' ', ' '.join(list('ABCDEFGH')))
        # 打印行名和棋盘
        for i in range(8):
            # print(board)
            print(str(i + 1), ' '.join(board[i]))
        if (not step_time) or (not total_time):
            # 棋盘初始化时展示的时间
            step_time = {"X": 0, "O": 0}
            total_time = {"X": 0, "O": 0}
            print("统计棋局: 棋子总数 / 每一步耗时 / 总时间 ")
            print("黑   棋: " + str(self.count('X')) + ' / ' + str(step_time['X']) + ' / ' + str(
                    total_time['X']))
            print("白   棋: " + str(self.count('O')) + ' / ' + str(step_time['O']) + ' / ' + str(
                    total_time['O']) + '\n')
        else:
            # 比赛时展示时间
            print("统计棋局: 棋子总数 / 每一步耗时 / 总时间 ")
            print("黑   棋: " + str(self.count('X')) + ' / ' + str(step_time['X']) + ' / ' + str(
                    total_time['X']))
            print("白   棋: " + str(self.count('O')) + ' / ' + str(step_time['O']) + ' / ' + str(
                    total_time['O']) + '\n')

    def count(self, color):
        """
        统计 color 一方棋子的数量。(O:白棋, X:黑棋, .:未落子状态)
        :param color: [O,X,.] 表示棋盘上不同的棋子
        :return: 返回 color 棋子在棋盘上的总数
        """
        count = 0
        for y in range(8):
            for x in range(8):
                if self._board[x][y] == color:
                    count += 1
        return count

    def get_winner(self):
        """
        判断黑棋和白旗的输赢，通过棋子的个数进行判断
        :return: 0-黑棋赢，1-白旗赢，2-表示平局，黑棋个数和白旗个数相等
        """
        # 定义黑白棋子初始的个数
        black_count, white_count = 0, 0
        for i in range(8):
            for j in range(8):
                # 统计黑棋棋子的个数
                if self._board[i][j] == 'X':
                    black_count += 1
                # 统计白旗棋子的个数
                if self._board[i][j] == 'O':
                    white_count += 1
        if black_count > white_count:
            # 黑棋胜
            return 0, black_count - white_count
        elif black_count < white_count:
            # 白棋胜
            return 1, white_count - black_count
        elif black_count == white_count:
            # 表示平局，黑棋个数和白旗个数相等
            return 2, 0

    def _move(self, action, color):
        """
        获取反转棋子的坐标
        :param action: 落子的坐标 可以是 D3 也可以是(2,3)
        :param color: [O,X,.] 表示棋盘上不同的棋子
        :return: 返回反转棋子的坐标列表
        """
        # 判断action 是不是字符串，如果是则转化为数字坐标
        if isinstance(action, str):
            action = self.board_num(action)
        # 获取 action 坐标的值
        x, y = action
        # 更改棋盘上 action 坐标处的状态，修改之后该位置属于 color[X,O,.]等三状态
        # print("color",color,action)
        self._board[x][y] = color

        # 将 action 转化为棋盘坐标
        action = self.num_board(action)
        return self._flip(action, color)

    def _flip(self, action, color):
        """
        反转棋子  返回反转棋子的坐标列表
        :param action: 落子坐标可以是 D3 也可以是(2,3)
        :param color: [O,X,.] 表示棋盘上棋子的状态
        :return:  返回反转棋子的坐标列表
        """
        # 储存反转棋子的数字坐标
        flipped_pos = []
        # 存储反转棋子的棋盘坐标。比如（2,3）--> D3
        flipped_pos_board = []
        # 判断action 是不是字符串，如果是则转化为数字坐标
        if isinstance(action, str):
            action = self.board_num(action)

        for line in self._get_lines(action):
            for i, p in enumerate(line):
                if self._board[p[0]][p[1]] == self.empty:
                    break
                elif self._board[p[0]][p[1]] == color:
                    flipped_pos.extend(line[:i])
                    break

        for p in flipped_pos:
            self._board[p[0]][p[1]] = color
            p_board = self.num_board(p)
            if p_board not in flipped_pos_board:
                flipped_pos_board.append(p_board)

        # return flipped_pos
        # 反转棋子打印的是棋盘坐标列表
        return flipped_pos_board

    def backpropagation(self, action, flipped_pos, color):
        """
        回溯
        :param action: 落子点的坐标
        :param flipped_pos: 反转棋子坐标列表
        :param color: 棋子的属性，[X,0,.]三种情况
        :return:
        """
        # 判断action 是不是字符串，如果是则转化为数字坐标
        if isinstance(action, str):
            action = self.board_num(action)

        self._board[action[0]][action[1]] = self.empty
        # 如果 color == 'X'，则 op_color = 'O';否则 op_color = 'X'
        if color == 'X':
            op_color = 'O'
        else:
            op_color = 'X'

        for p in flipped_pos:
            # 判断一下p是不是数字坐标
            # 判断action 是不是字符串，如果是则转化为数字坐标
            if isinstance(p, str):
                p = self.board_num(p)
            self._board[p[0]][p[1]] = op_color

    def _get_lines(self, action):
        """
        根据 action 落子坐标获取 8 个方向的下标数组
        :param action: 落子坐标
        :return: 8 个方向坐标
        """
        # 判断action 是不是字符串，如果是则转化为数字坐标
        if isinstance(action, str):
            action = self.board_num(action)
        row, col = action
        # 生成棋盘坐标
        board_coord = [(i, j) for i in range(8) for j in range(8)]  # 棋盘坐标

        ix = row * 8 + col
        # 获取ix // 8 整除，取余
        r, c = ix // 8, ix % 8
        left = board_coord[r * 8:ix]  # 要反转
        right = board_coord[ix + 1:(r + 1) * 8]
        top = board_coord[c:ix:8]  # 要反转
        bottom = board_coord[ix + 8:8 * 8:8]

        if r <= c:
            lefttop = board_coord[c - r:ix:9]  # 要反转
            rightbottom = board_coord[ix + 9:(7 - (c - r)) * 8 + 7 + 1:9]
        else:
            lefttop = board_coord[(r - c) * 8:ix:9]  # 要反转
            rightbottom = board_coord[ix + 9:7 * 8 + (7 - (c - r)) + 1:9]

        if r + c <= 7:
            leftbottom = board_coord[ix + 7:(r + c) * 8:7]
            righttop = board_coord[r + c:ix:7]  # 要反转
        else:
            leftbottom = board_coord[ix + 7:7 * 8 + (r + c) - 7 + 1:7]
            righttop = board_coord[((r + c) - 7) * 8 + 7:ix:7]  # 要反转

        # 有四个要反转，方便判断
        left.reverse()
        top.reverse()
        lefttop.reverse()
        righttop.reverse()
        lines = [left, top, lefttop, righttop, right, bottom, leftbottom, rightbottom]
        # 返回数字坐标
        return lines

    def _can_fliped(self, action, color):
        """
        检测，位置是否有子可翻
        :param action: 下子位置
        :param color: [X,0,.] 棋子状态
        :return:
        """
        # 判断action 是不是字符串，如果是则转化为数字坐标
        if isinstance(action, str):
            action = self.board_num(action)

        flipped_pos = []

        for line in self._get_lines(action):
            for i, p in enumerate(line):
                if self._board[p[0]][p[1]] == self.empty:
                    break
                elif self._board[p[0]][p[1]] == color:
                    flipped_pos.extend(line[:i])
                    break
        # 检测位置有棋子可以反转，则返回 True，否则 False
        if len(flipped_pos) > 0:
            return True
        else:
            return False

    def get_legal_actions(self, color):
        """
        按照黑白棋的规则获取棋子的合法走法
        :param color: 不同颜色的棋子，X-黑棋，O-白棋
        :return: 生成合法的落子坐标，用list()方法可以获取所有的合法坐标
        """
        # 表示棋盘坐标点的8个不同方向坐标，比如方向坐标[0][1]则表示坐标点的正上方。
        # direction = [(-1, -1), ]
        direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        # 如果 color == 'X'，则 op_color = 'O';否则 op_color = 'X'
        if color == 'X':
            # 如果color代表黑棋一方，则op_color代表白棋一方; 否则color代表白棋一方，op_color代表黑棋一方
            op_color = 'O'
        else:
            op_color = 'X'

        # 统计 op_color 一方邻近的未落子状态的位置
        op_color_near_points = []

        board = self._board
        for i in range(8):
            # i 是行数，从0开始，j是列数，也是从0开始
            for j in range(8):
                # 判断棋盘[i][j]位子棋子的属性，如果是op_color，则继续进行下一步操作，
                # 否则继续循环获取下一个坐标棋子的属性
                if board[i][j] == op_color:
                    # dx，dy 分别表示[i][j]坐标在行、列方向上的步长，direction 表示方向坐标
                    for dx, dy in direction:
                        x, y = i + dx, j + dy
                        # 表示x、y坐标值在合理范围，棋盘坐标点board[x][y]为未落子状态，
                        # 而且（x,y）不在op_color_near_points 中，统计对方未落子状态位置的列表才可以添加该坐标点
                        if 0 <= x <= 7 and 0 <= y <= 7 and board[x][y] == self.empty and (
                                x, y) not in op_color_near_points:
                            op_color_near_points.append((x, y))
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        for p in op_color_near_points:
            if self._can_fliped(p, color):
                # 判断p是不是数字坐标，如果是则返回棋盘坐标
                if p[0] in l and p[1] in l:
                    p = self.num_board(p)
                yield p

    def board_num(self, action):
        """
        棋盘坐标转化为数字坐标
        :param action:棋盘坐标，比如A1
        :return:数字坐标，比如 A1 --->(0,0)
        """
        row, col = str(action[1]).upper(), str(action[0]).upper()
        if row in '12345678' and col in 'ABCDEFGH':
            # 坐标正确
            x, y = '12345678'.index(row), 'ABCDEFGH'.index(col)
            return x, y

    def num_board(self, action):
        """
        数字坐标转化为棋盘坐标
        :param action:数字坐标 ,比如(0,0)
        :return:棋盘坐标，比如 （0,0）---> A1
        """
        row, col = action
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        if col in l and row in l:
            return chr(ord('A') + col) + str(row + 1)


# 测试
if __name__ == '__main__':
    board = Board()  # 棋盘初始化
    # 其中step_time 表示每一步双方落子时间，total_time 表示双方从开始到现在的落子总时间，单位是秒
    board.display()
