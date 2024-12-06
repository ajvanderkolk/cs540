import random
from copy import deepcopy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        return False

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = True   # detect drop phase

        bcount = sum((i.count('b') for i in state))
        rcount = sum((i.count('r') for i in state))
        if bcount >= 4 and rcount >= 4:
            drop_phase = False

        move = []
        

        if not drop_phase:
            # choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            max, state_max = self.max_value(state, 0)
            # orig_pos = None
            # new_pos = None
            # for i in range(5):
            #     for j in range(5):
            #         if state[i][j] != state_max[i][j]:
            #             if state[i][j] == self.my_piece:
            #                 orig_pos = (i, j)
            #             else:
            #                 new_pos = (i, j)
            # move.append([new_pos, orig_pos])
            return state_max

        # select an unoccupied space randomly
        # implement a minimax algorithm to play better
        def drop_phase_heuristic(state, player):
            """
            Heuristic for the drop phase of the game.

            :param state: 2D list representing the game board.
            :param player: The player's piece ('b' or 'r').
            :return: Tuple (best_move_row, best_move_col).
            """
            opponent = 'r' if player == 'b' else 'b'
            rows, cols = len(state), len(state[0])

            def is_valid_move(row, col):
                return 0 <= row < rows and 0 <= col < cols and state[row][col] == ' '

            def check_in_a_row(board, player, num):
                """Returns positions where 'player' can make 'num' in a row."""
                moves = []
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal, anti-diagonal
                for r in range(rows):
                    for c in range(cols):
                        if board[r][c] != player:
                            continue
                        for dr, dc in directions:
                            count = 0
                            last_empty = None
                            for i in range(-num + 1, num):  # Check around this position
                                nr, nc = r + dr * i, c + dc * i
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if board[nr][nc] == player:
                                        count += 1
                                    elif board[nr][nc] == ' ':
                                        last_empty = (nr, nc)
                                    else:
                                        break
                            if count == num - 1 and last_empty:
                                moves.append(last_empty)
                return moves

            def first_valid_move(preferred_moves):
                """Return the first valid move from the preferred moves list."""
                for move in preferred_moves:
                    if is_valid_move(*move):
                        return move
                return None

            # Preferred positions for initial plays
            center = [(2, 2)]  # C3 (0-indexed as row 2, col 2)
            secondary = [(1, 1), (3, 1), (1, 3), (3, 3)]  # B2, D2, B4, D4 (0-indexed)

            # Ply 1: Play on C3 if available
            move = first_valid_move(center)
            if move:
                return move

            # Reply 1: If opponent can make two in a row, block it; otherwise play on secondary positions
            opponent_two = check_in_a_row(state, opponent, 2)
            if opponent_two:
                return opponent_two[0]
            move = first_valid_move(secondary)
            if move:
                return move

            # Ply 2: If possible, make two in a row; else block opponent from making three in a row
            my_two = check_in_a_row(state, player, 2)
            if my_two:
                return my_two[0]
            opponent_three = check_in_a_row(state, opponent, 3)
            if opponent_three:
                return opponent_three[0]

            # Reply 2: Avoid opponent making three in a row if necessary, else make two in a row
            opponent_three = check_in_a_row(state, opponent, 3)
            if opponent_three:
                return opponent_three[0]
            my_two = check_in_a_row(state, player, 2)
            if my_two:
                return my_two[0]

            # Ply 3: If possible, make three in a row; else block opponent from making four in a row
            my_three = check_in_a_row(state, player, 3)
            if my_three:
                return my_three[0]
            opponent_four = check_in_a_row(state, opponent, 4)
            if opponent_four:
                return opponent_four[0]

            # Reply 3: Avoid opponent making four in a row if necessary, else make three in a row
            opponent_four = check_in_a_row(state, opponent, 4)
            if opponent_four:
                return opponent_four[0]
            my_three = check_in_a_row(state, player, 3)
            if my_three:
                return my_three[0]

            # Ply 4: If possible, make four in a row; else block opponent from making four in a row
            my_four = check_in_a_row(state, player, 4)
            if my_four:
                return my_four[0]
            opponent_four = check_in_a_row(state, opponent, 4)
            if opponent_four:
                return opponent_four[0]

            # Reply 4: Make four in a row or block if needed
            my_four = check_in_a_row(state, player, 4)
            if my_four:
                return my_four[0]

            # Default: Return first available move
            for r in range(rows):
                for c in range(cols):
                    if is_valid_move(r, c):
                        return r, c
            
        move = move.insert(0, drop_phase_heuristic(state, self.my_piece))

        # good_spots = [(2,2), (2,1), (1,2), (1,1), (2,3), (3,2), (3,3), (1,3), (3,1)]
        # for spot in good_spots:
        #     if state[spot[0]][spot[1]] == ' ':
        #         move.insert(0, spot)
        #         return move
        if move == None:
            move = []
            (row, col) = (random.randint(0,4), random.randint(0,4))
            while not state[row][col] == ' ':
                (row, col) = (random.randint(0,4), random.randint(0,4))
            # ensure the destination (row,col) tuple is at the beginning of the move list
            move.insert(0, (row, col))
        return move

    def succ(self, state, piece):
        """ Generate the successors of the current state

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

        Returns:
            list of lists of tuples: [(row, col), (source_row, source_col)]
        """
        succ = []

        # drop phase successors
        drop_phase = True
        b_count = sum((i.count('b') for i in state))
        r_count = sum((i.count('r') for i in state))
        if b_count >= 4 and r_count >= 4:
            drop_phase = False
        
        # move phase successors
        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == piece:
                        # check and store all possible moves from this piece
                        if i > 0:
                            # upper left
                            if j > 0 and state[i-1][j-1] == ' ':
                                next_state = [(i-1,j-1),(i,j)]
                                succ.append(next_state)
                            # upper middle
                            if state[i-1][j] == ' ':
                                next_state = [(i-1,j),(i,j)]
                                succ.append(next_state)
                            # upper right
                            if j < 4 and state[i-1][j+1] == ' ':
                                next_state = [(i-1,j+1),(i,j)]
                                succ.append(next_state)
                        # middle left
                        if j > 0 and state[i][j-1] == ' ':
                                next_state = [(i,j-1),(i,j)]
                                succ.append(next_state)
                        # middle right
                        if j < 4 and state[i][j+1] == ' ':
                                next_state = [(i,j+1),(i,j)]
                                succ.append(next_state)
                        if i < 4:
                            # lower left
                            if j > 0 and state[i+1][j-1] == ' ':
                                next_state = [(i+1,j-1),(i,j)]
                                succ.append(next_state)
                            # lower middle
                            if state[i+1][j] == ' ':
                                next_state = [(i+1,j),(i,j)]
                                succ.append(next_state)
                            # lower right
                            if j < 4 and state[i+1][j+1] == ' ':
                                next_state = [(i+1,j+1),(i,j)]
                                succ.append(next_state)
            return succ

        # drop phase successors
        for i in range(5):
            for j in range(5):
                if state[i][j] == ' ':
                    next_state = (i,j)
                    succ.append(next_state)

        return succ

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            if state[i][i] != ' ' and state[i][i] == state[i+1][i+1] == state[i+2][i+2] == state[i+3][i+3]:
                return 1 if state[i][i]==self.my_piece else -1
            if i == 0 and state[i][i+1] != ' ' and state[i][i+1] == state[i+1][i+2] == state[i+2][i+3] == state[i+3][i+4]:
                return 1 if state[i][i]==self.my_piece else -1
            if i == 0 and state[i+1][i] != ' ' and state[i+1][i] == state[i+2][i+1] == state[i+3][i+2] == state[i+4][i+3]:
                return 1 if state[i][i]==self.my_piece else -1

        # check / diagonal wins
        for i in range(5):
            if i < 2 and state[i][4-i] != ' ' and state[i][4-i] == state[i-1][3-i] == state[i-2][2-i] == state[i-3][1-i]:
                return 1 if state[i][4-i]==self.my_piece else -1
            if i > 2 and state[i][3-i] != ' ' and state[i][3-i] == state[i-1][i-2] == state[i-2][i-1] == state[i-3][i]:
                return 1 if state[i][3-i]==self.my_piece else -1

        # check box wins
        for i in range(1,4):
            if state[i][i] != ' ' and (
                state[i][i] == state[i-1][i-1] == state[i-1][i] == state[i][i-1] or
                state[i][i] == state[i-1][i] == state[i-1][i+1] == state[i][i+1] or
                state[i][i] == state[i][i+1] == state[i+1][i+1] == state[i+1][i] or
                state[i][i] == state[i+1][i] == state[i+1][i-1] == state[i][i-1]):
                return 1 if state[i][i]==self.my_piece else -1

        return 0 # no winner yet

    def min_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state), state
        
        if depth >= 3:
            return self.heuristic_game_value(state, self.opp)
        else:
            min = float('Inf')

            for move in self.succ(state, self.opp):
                value = self.max_value(self.update_state(state, self.opp, move), depth+1)

                if value[0] < min:
                    min = value[0]
                    min_state = move
            return (min, min_state)
    
    def max_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state), state
        
        if depth >= 3:
            return self.heuristic_game_value(state)
        else:
            max = float('-Inf')

            for move in self.succ(state, self.my_piece):
                value = self.min_value(self.update_state(state, self.my_piece, move), depth+1)

                if value[0] > max:
                    max = value[0]
                    max_state = move
            return (max, max_state)
    
    def update_state(self, state, piece, move):
        new_state = deepcopy(state)
        if type(move) == tuple:
            new_state[move[0]][move[1]] = piece
        if type(move) == list:
            new_state[move[0][0]][move[0][1]] = piece
            new_state[move[1][0]][move[1][1]] = ' '
        return new_state
    
    def heuristic_game_value(self, state, piece):
        mine = piece
        oppo = 'r' if piece == 'b' else 'b'

        def count_max_occurrences(board, player):
            max_count = 0
            for row in board:
                count = row.count(player)
                max_count = max(max_count, count)
            return max_count
        
        mymax = max(count_max_occurrences(state, mine), count_max_occurrences(zip(*state), mine))
        oppmax = max(count_max_occurrences(state, oppo), count_max_occurrences(zip(*state), oppo))

        def count_diagonal(board, player):
            max_count = 0
            for row in range(len(board) - 3):
                for col in range(len(board[row]) - 3):
                    count = sum(board[row + i][col + i] == player for i in range(4))
                    max_count = max(max_count, count)
            return max_count
        
        mymax = max(mymax, count_diagonal(state, mine), count_diagonal(state[::-1], mine))
        oppmax = max(oppmax, count_diagonal(state, oppo), count_diagonal(state[::-1], oppo))

        def count_2x2(board, player):
            max_count = 0
            for row in range(len(board) - 1):
                for col in range(len(board[row]) - 1):
                    count = sum(board[row + i][col + j] == player for i in range(2) for j in range(2))
                    max_count = max(max_count, count)
            return max_count

        mymax = max(mymax, count_2x2(state, mine))
        oppmax = max(oppmax, count_2x2(state, oppo))

        if mymax == oppmax:
            return 0, state
        elif mymax >= oppmax:
            return mymax / 6, state
        else:
            return -oppmax / 6, state



############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()