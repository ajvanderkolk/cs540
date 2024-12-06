import random

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
        self.depth_limit = 2


    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        return False
    

    def get_drop_phase(self, state):
        """ Detects if the game is in the drop phase or not

        Args:
            state (list of lists)

        Returns:
            bool: True if the game is in the drop phase, False otherwise
        """
        count = 0
        for row in state:
            for cell in row:
                if cell != ' ':
                    count += 1
        return count < 8
    
    def succ(self, state):
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
        if self.get_drop_phase(state):
            for i in range(5): 
                for j in range(5):
                    if state[i][j] == ' ':
                        succ.append((i,j))
            return succ
        
        # move phase successors
        else:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
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

        move = []
        drop_phase = self.get_drop_phase(state)   # detect drop phase

        if drop_phase:
            # try better spots
            good_spots = [(2,2), (2,1), (1,2)]
            for spot in good_spots:
                if state[spot[0]][spot[1]] == ' ':
                    move.insert(0, spot)
                    return move

            # select an unoccupied space randomly
            (row, col) = (random.randint(0,4), random.randint(0,4))
            while not state[row][col] == ' ':
                (row, col) = (random.randint(0,4), random.randint(0,4))

            # ensure the destination (row,col) tuple is at the beginning of the move list
            move.insert(0, (row, col))

        if not drop_phase:
            # select a piece to move and a destination for that piece
            move = self.max_value(state, 0, -1, 1)        
            
        return move


    def max_value(self, state, depth, alpha, beta):
        game_val = self.game_value(state)
        # if game_val == 1 or game_val == -1:
        #     return game_val
        
        # if depth == self.depth_limit:
        #     return self.heuristic_game_value(state)
        
        # grade successors
        max = 0
        best_move = None
        successors = self.succ(state)
        for s in successors:
            f = s[0]
            t = s[1]
            s[f[0]][f[1]] = ' '
            s[t[0]][t[1]] = self.my_piece
            value = self.heuristic_game_value(s)
            if value > max:
                max = value
                best_move = s

        # if best_move == None:
        #     return self.heuristic_game_value(state)
        return best_move


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
    
    
    def heuristic_game_value(self, state):
        """ Checks the current board status for a non-win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            
        """
        # check for the game status
        state_value = self.game_value(state)
        
        if state_value != 0:
            return state_value
        
        # initialize the weight for the piece
        general_weight = 0.1
        center_bonus = 0.2
        special_position_bonus = 0.1
        
        # initialzie the player score
        player_score = 0.0
        ai_score = 0.0
        
        # traverse the game board to compute the score
        for row in range(5):
            for col in range(5):
                # check for the player score
                if state[row][col] == self.my_piece:
                    player_score += general_weight

                    # additional values for center position
                    if row == 2 and col == 2:
                        player_score += center_bonus

                    # additional values for special position
                    if (row == 2 and col == 1) or (row == 1 and col == 2):
                        player_score += special_position_bonus

                # check for the opposite score
                elif state[row][col] == self.opp:
                    ai_score += general_weight
                    
                     # additional values for center position
                    if row == 2 and col == 2:
                        ai_score += center_bonus

                    # additional values for special position
                    if (row == 2 and col == 1) or (row == 1 and col == 2):
                        ai_score += special_position_bonus

        # compute heuristic value
        heuristic_value = player_score - ai_score

        return heuristic_value
    

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