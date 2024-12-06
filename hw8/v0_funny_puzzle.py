import heapq

def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for idx, value in enumerate(from_state):
        if value != 0:  # Ignore empty spaces
            goal_idx = to_state.index(value)
            distance += abs(idx // 3 - goal_idx // 3) + abs(idx % 3 - goal_idx % 3)
    return distance

def print_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid succ_states in the puzzle. 
    """

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)

    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_get_manhattan_distance(succ_state,goal_state)))


def get_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid succ_states in the puzzle (don't forget to sort the result as done below). 
    """
   
    succ_states = []
    zero_idx = state.index(0)  # Find the single empty space (one zero)
    row, col = zero_idx // 3, zero_idx % 3
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state = state[:]
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            succ_states.append(new_state)
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # This is a format helper，which is only designed for format purpose.
    # define "solvable_condition" to check if the puzzle is really solvable
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute "max_length", it might be useful in debugging
    # it can help to avoid any potential format issue.

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)

    # if not solvable_condition:
    #     print(False)
    #     return
    # else:
    #     print(True)

    # for state_info in state_info_list:
    #     current_state = state_info[0]
    #     h = state_info[1]
    #     move = state_info[2]
    #     print(current_state, "h={}".format(h), "moves: {}".format(move))
    # print("Max queue length: {}".format(max_length))

    pq = []
    visited = set()
    parent_map = {}
    heapq.heappush(pq, (0, state, (0, get_manhattan_distance(state, goal_state), None)))

    while pq:
        cost, current_state, (g, h, parent) = heapq.heappop(pq)
        if tuple(current_state) in visited:
            continue
        visited.add(tuple(current_state))
        parent_map[tuple(current_state)] = (g, h, parent)
        if current_state == goal_state:
            print("True")
            # Reconstruct path
            path = []
            state = tuple(current_state)
            while state is not None:
                g, h, parent = parent_map[state]
                path.append((list(state), h, g))
                state = parent
            path.reverse()
            for state, h, moves in path:
                print(f"{state} h={h} moves: {moves}")
            print(f"Max queue length: {len(pq) + len(visited)}")
            return
        for succ in get_succ(current_state):
            if tuple(succ) not in visited:
                g_new = g + 1
                h_new = get_manhattan_distance(succ, goal_state)
                heapq.heappush(pq, (g_new + h_new, succ, (g_new, h_new, tuple(current_state))))
    print("False")

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    #solve([2,5,1,4,0,6,7,0,3])
    solve([4,3,0,5,1,6,7,2,0])
    print()
