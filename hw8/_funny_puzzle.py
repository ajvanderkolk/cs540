
import heapq

# Heuristic function: Manhattan distance
def manhattan_distance(state, goal):
    distance = 0
    for idx, value in enumerate(state):
        if value != 0:  # Ignore empty spaces
            goal_idx = goal.index(value)
            distance += abs(idx // 3 - goal_idx // 3) + abs(idx % 3 - goal_idx % 3)
    return distance

# Generate successors
def generate_successors(state):
    successors = []
    zero_idx = state.index(0)  # Find the single empty space (one zero)
    row, col = zero_idx // 3, zero_idx % 3
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state = state[:]
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            successors.append(new_state)
    return sorted(successors)

# Print successors with heuristic values
def print_succ(state):
    goal_state = sorted([x for x in state if x != 0]) + [0]
    successors = generate_successors(state)
    for succ in successors:
        print(f"{succ} h={manhattan_distance(succ, goal_state)}")

# A* solve function
def solve(initial_state):
    goal_state = sorted([x for x in initial_state if x != 0]) + [0]
    pq = []
    visited = set()
    parent_map = {}
    heapq.heappush(pq, (0, initial_state, (0, manhattan_distance(initial_state, goal_state), None)))

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
        for succ in generate_successors(current_state):
            if tuple(succ) not in visited:
                g_new = g + 1
                h_new = manhattan_distance(succ, goal_state)
                heapq.heappush(pq, (g_new + h_new, succ, (g_new, h_new, tuple(current_state))))
    print("False")
