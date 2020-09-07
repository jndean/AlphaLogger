


def move_idx_to_tuple(idx):
    y = idx // 50
    x = (idx // 10) % 5
    action = idx % 10
    return (y, x, action, 0, 0)

def move_tuple_to_idx(move):
    y, x, action, _, _ = move
    return (50 * y) + (10 * x) + action

def stringify_move(move):
    if not isinstance(move, tuple):
        move = move_idx_to_tuple(move)
    y, x, action, p_y, p_x = move
    act_str = [
    'Chop Up', 'Chop Left', 'Chop Right', 'Chop Down',
    'Plant Up', 'Plant Left', 'Plant Right', 'Plant Down',
    f'Protest({p_y}, {p_x})', 'Pass'
    ][action]
    return f'Move({y}, {x}), {act_str}'
