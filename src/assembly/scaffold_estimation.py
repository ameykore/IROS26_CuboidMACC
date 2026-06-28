import numpy as np

from typing import Tuple, List, Iterable
from numpy.typing import NDArray

from copy import copy
from functools import lru_cache

from assembly import block_utils


# ============================================================
# UPDATED: cached neighbor generation
#   - Original generate_d_neighbors is a generator used in tight loops.
#   - We cache the (x,y,d,x_dim,y_dim,recursive) -> tuple(coords) result.
#   - This is especially helpful because the same distance patterns repeat
#     heavily across groups/heights in heterogeneous cases.
# ============================================================

@lru_cache(maxsize=None)
def _cached_d_neighbors(x: int, y: int, distance: int, x_dim: int, y_dim: int, recursive: bool) -> Tuple[Tuple[int, int], ...]:
    """
    Cached version of generate_d_neighbors.
    Returns a tuple of (nx, ny) pairs.
    """
    if distance <= 0:
        return tuple()

    out: List[Tuple[int, int]] = []

    candidate = (x - distance, y)
    if block_utils.in_bounds(*candidate, 0, x_dim, y_dim, 1):
        out.append(candidate)

    for dx in range(1 - distance, distance):
        mag_dy = distance - abs(dx)
        candidate = (x + dx, y + mag_dy)
        if block_utils.in_bounds(*candidate, 0, x_dim, y_dim, 1):
            out.append(candidate)
        candidate = (x + dx, y - mag_dy)
        if block_utils.in_bounds(*candidate, 0, x_dim, y_dim, 1):
            out.append(candidate)

    candidate = (x + distance, y)
    if block_utils.in_bounds(*candidate, 0, x_dim, y_dim, 1):
        out.append(candidate)

    if recursive and distance > 2:
        out.extend(_cached_d_neighbors(x, y, distance - 2, x_dim, y_dim, recursive))

    return tuple(out)


def generate_d_neighbors(x, y, distance, x_dim, y_dim, recursive=True):
    """
    Generate x,y coordinates that have a length-d path from the query

    if recursive is True, will additionally return cells whose shortest path is <d
    but have an indirect length-d path

    UPDATED: uses cached neighbor lists.
    """
    for c in _cached_d_neighbors(int(x), int(y), int(distance), int(x_dim), int(y_dim), bool(recursive)):
        yield c


class ScaffoldInfo:
    groups: NDArray
    group2support: List[NDArray]
    sizes: List[NDArray]
    scaffold_val: NDArray
    useful_support: List[NDArray]

    def __init__(self, groups, group2support, sizes, scaffold_val, useful_support):
        """
        groups : (z_dim,x_dim,y_dim) int np array
            (z,x,y) is the id number of the group that goal block cell belongs to.
            Non-goal cells get 0. Group Ids are reused at each z value.
        group2support : List[(num groups at height, z_dim-1,2,x_dim,y_dim) boolean arrays]
            entry i contains, for each group at height i, an indicator array showing
            [scaffolding options in interior of shadow, scaffolding options on surface of shadow]
        sizes : List[NDArray]
            entry i contains, for each group at height i, the integer number of elements in that group
        scaffold_val : (z_dim,z_dim-1,x_dim,y_dim) int8 np array
            [h, z, x, y] is the value of a support block at x,y,z for goals at height h
        useful_support : length z_dim List[(num_groups,z_dim-1)] np int8 arrays
            useful_support[h][i,z] is True if a scaffold at height z is useful for the ith group at height h
        """
        self.groups = groups
        self.group2support = group2support
        self.sizes = sizes
        self.scaffold_val = scaffold_val
        self.useful_support = useful_support

    @classmethod
    def from_occupancy_grids(cls, current_occupancy_grid: NDArray, goal_occupancy_grid: NDArray):
        groups = find_groups(current_occupancy_grid.shape, goal_occupancy_grid, current_occupancy_grid)
        group2support, sizes = find_group_support(current_occupancy_grid.shape, groups)
        scaffold_val, useful_support = cast_scaffold_value(
            current_occupancy_grid.shape, goal_occupancy_grid, group2support, sizes, current_occupancy_grid
        )
        return cls(groups, group2support, sizes, scaffold_val, useful_support)

    # ============================================================
    # UPDATED: avoid deepcopy(self.useful_support) every expansion
    #   - deepcopy was expensive and happens for *every* high-level node expansion
    #   - We only copy the list + the specific arrays that update_scaffold_value mutates
    # ============================================================
    @staticmethod
    def _copy_useful_support_for_update(useful_support: List[NDArray], z: int, is_goal: bool) -> List[NDArray]:
        """
        update_scaffold_value mutates:
          - useful_support[height][i, support_height] for height in range(z+1, z_dim)
            and support_height in {z} or {z,z-1} if is_goal and z>0
          - and if is_goal: it replaces useful_support[z] entirely

        So: shallow-copy list, and copy arrays for heights >= z (conservative and safe).
        """
        new_list = list(useful_support)  # shallow copy
        for h in range(z, len(new_list)):
            new_list[h] = new_list[h].copy()
        return new_list

    def update_by_action(self, action: Tuple[int, int, int, int, int, int], resulting_occupancy_grid: NDArray, goal):
        """
        Create a new ScaffoldingInfo reflecting taking an action in the world that the current ScaffoldingInfo corresponds to

        Parameters:
            action : Tuple[int,int,int,int,int,int]
                [-1 if removal|1 if addition, block length, 0 if horizontal|1 if vertical, block x, block y, block z]
            resulting_occupancy_grid : np boolean array, shape=world_size
                which cells are occupied after completing action
            goal : problem_node.Assembly_Node
                describes the desired world state

        Returns:
            valid : bool
                False if any of the cal_support_val calls return invalid
            child_info : ScaffoldingInfo
                new ScaffoldingInfo after taking action
        """
        action_type, block_length, rotation, x, y, z = action
        is_goal = action in goal.curr_node
        is_goal_off_ground = is_goal and z > 0

        if is_goal_off_ground:
            new_groups = self.groups.copy()
            update_groups(action, resulting_occupancy_grid.shape, new_groups)

            # deepcopy would be wasteful: list container changes at only one z slice
            new_group2support = copy(self.group2support)
            new_group_sizes = copy(self.sizes)
            new_group2support[z], new_group_sizes[z] = compute_group_support(z, resulting_occupancy_grid.shape, new_groups[z])
        else:
            new_groups = self.groups
            new_group2support = self.group2support
            new_group_sizes = self.sizes

        new_scaffold_val = self.scaffold_val.copy()
        new_useful_support = self._copy_useful_support_for_update(self.useful_support, z, is_goal)

        valid = update_scaffold_value(
            action,
            is_goal,
            resulting_occupancy_grid,
            goal.reachability.occupancy_grid,
            new_group2support,
            new_group_sizes,
            new_scaffold_val,
            new_useful_support,
        )
        return valid, ScaffoldInfo(new_groups, new_group2support, new_group_sizes, new_scaffold_val, new_useful_support)

    def estimate_missing_scaffolding(self, x_dim, y_dim, z_dim):
        return cal_goal_val((x_dim, y_dim, z_dim), self.group2support, self.useful_support, self.scaffold_val)


def find_groups(world_size: Tuple[int, int, int], goal_occupancy_grid: NDArray, current_occupancy_grid: NDArray):
    """
    find the groups of unplaced goal blocks that are adjacent and at the same height as each other
    """
    groups = []
    for level in range(world_size[2]):
        unfinished_goals = np.logical_and(goal_occupancy_grid[:, :, level], np.logical_not(current_occupancy_grid[:, :, level]))
        group_map = np.zeros((world_size[0], world_size[1]), dtype=np.int8)
        gid = 1
        for x in range(world_size[0]):
            for y in range(world_size[1]):
                if unfinished_goals[x, y]:
                    this_group_map = np.zeros((world_size[0], world_size[1]), dtype=np.bool_)
                    connect_goals(world_size, unfinished_goals, x, y, this_group_map)
                    group_map += gid * this_group_map  # puts the integer group id in the cells belonging to that group
                    gid += 1
        groups.append(group_map)
    groups = np.stack(groups, axis=0)
    return groups


def update_groups(action: Tuple[int, int, int, int, int, int], world_size: Tuple[int, int, int], groups: NDArray):
    x_dim, y_dim, z_dim = world_size
    action_type, block_length, rotation, x, y, z = action
    group_map = groups[z]
    if action_type == 1:
        # addition
        gid = group_map[x, y]
        new_id = np.max(group_map) + 1
        group_map[x, y] = 0
        for nx, ny in block_utils.horizontal_neighbors(x, y, x_dim, y_dim):
            this_group_map = np.zeros((x_dim, y_dim), dtype=np.bool_)
            if group_map[nx, ny] == gid:
                connect_goals((x_dim, y_dim, z_dim), group_map, nx, ny, this_group_map, val=gid)
                if np.any(group_map == gid):
                    group_map += this_group_map * new_id
                    new_id += 1
                else:
                    group_map += this_group_map * gid
                    break
    else:
        raise ValueError("Tried to remove a goal block")


def connect_goals(world_size: Tuple[int, int, int], workspace: NDArray, query_x: int, query_y: int, this_group_map: NDArray, val=True, new_val=False):
    """
    connect neighboring goals into a group by marking this_group_map True and workspace to new_val for every cell in the group
    """
    if workspace[query_x, query_y] != val:
        return
    workspace[query_x, query_y] = new_val
    this_group_map[query_x, query_y] = True
    for x, y in block_utils.horizontal_neighbors(query_x, query_y, world_size[0], world_size[1]):
        connect_goals(world_size, workspace, x, y, this_group_map, val, new_val)


def find_group_support(world_size: Tuple[int, int, int], groups: NDArray):
    group2support = []
    sizes = []
    for level in range(world_size[2]):
        group_map = groups[level]
        support_map, group_sizes = compute_group_support(level, world_size, group_map)
        group2support.append(support_map)
        sizes.append(group_sizes)
    return group2support, sizes


def compute_group_support(level: int, world_size: tuple[int, int, int], group_map: NDArray):
    gids = np.unique(group_map)[1:]
    number_of_groups = len(gids)
    group_sizes = np.zeros(number_of_groups, dtype=np.int8)
    support_map = np.zeros((number_of_groups, world_size[2] - 1, 2, world_size[0], world_size[1]), dtype=np.bool_)
    for i, g in enumerate(gids):
        g_map = group_map == g
        relevant_columns = np.transpose(np.nonzero(g_map))
        group_sizes[i] = len(relevant_columns)
        s_map = support_map[i]
        for support_height in range(level):
            for x, y in relevant_columns:
                for support_cell in generate_d_neighbors(x, y, level - support_height - 2, world_size[0], world_size[1], True):
                    if not g_map[support_cell[0], support_cell[1]]:
                        s_map[support_height, 0, support_cell[0], support_cell[1]] = True
            for x, y in relevant_columns:
                for support_cell in generate_d_neighbors(x, y, level - support_height, world_size[0], world_size[1], False):
                    if not g_map[support_cell[0], support_cell[1]]:
                        if not s_map[support_height, 0, support_cell[0], support_cell[1]]:
                            s_map[support_height, 1, support_cell[0], support_cell[1]] = True
    return support_map, group_sizes


def cast_scaffold_value(world_size: Tuple[int, int, int], goal_occupancy_grid: NDArray, group2support: List[NDArray], group_sizes: List[NDArray], current_occupancy_grid: NDArray):
    scaffold_val = np.zeros((world_size[2], world_size[2] - 1, world_size[0], world_size[1]), dtype=np.int8)
    useful_support = []
    for height in range(world_size[2]):
        val = scaffold_val[height]
        supports = group2support[height]
        sizes = group_sizes[height]
        number_of_groups = supports.shape[0]
        useful = np.zeros((number_of_groups, world_size[2] - 1), dtype=np.int8)
        for i in range(number_of_groups):
            for support_height in range(height):
                support = supports[i, support_height]
                _, u, v = cal_support_val(support_height, support, current_occupancy_grid, goal_occupancy_grid, height, sizes[i])
                val[support_height] += v
                useful[i, support_height] = u
        useful_support.append(useful)
    return scaffold_val, useful_support


def update_scaffold_value(
    action: Tuple[int, int, int, int, int, int],
    is_goal: bool,
    current_occupancy_grid: NDArray,
    goal_occupancy_grid: NDArray,
    group2support: List[NDArray],
    group_sizes: List[NDArray],
    scaffold_val: NDArray,
    useful_support: List[NDArray],
):
    """
    update scaffold value and the map of useful supports based on an action taken
    """
    action_type, block_length, rotation, x, y, z = action
    x_dim, y_dim, z_dim = goal_occupancy_grid.shape

    # process goals above the action
    if is_goal and z > 0:
        zs = [z, z - 1]
    else:
        zs = [z]

    for height in range(z + 1, z_dim):
        useful = useful_support[height]
        support = group2support[height]
        size = group_sizes[height]
        number_of_groups = useful.shape[0]
        for support_height in zs:
            val = np.zeros((x_dim, y_dim), dtype=np.int8)
            for i in range(number_of_groups):
                this_group_support = support[i, support_height]
                valid, u, v = cal_support_val(support_height, this_group_support, current_occupancy_grid, goal_occupancy_grid, height, size[i])
                if not valid:
                    return False
                val += v
                useful[i, support_height] = u
            scaffold_val[height, support_height] = val

    if is_goal:
        val = np.zeros((z_dim - 1, x_dim, y_dim), dtype=np.int8)
        support = group2support[z]
        size = group_sizes[z]
        number_of_groups = support.shape[0]
        useful = np.zeros((number_of_groups, z_dim - 1), dtype=np.int8)
        for i in range(number_of_groups):
            for support_height in range(z):
                this_group_support = support[i, support_height]
                valid, u, v = cal_support_val(support_height, this_group_support, current_occupancy_grid, goal_occupancy_grid, z, size[i])
                if not valid:
                    return False
                val[support_height] += v
                useful[i, support_height] = u
        scaffold_val[z] = val
        useful_support[z] = useful
    return True


def cal_support_val(
    support_height: int,
    support: NDArray,
    current_occupancy_grid: NDArray,
    goal_occupancy_grid: NDArray,
    goal_height: int,
    number_of_group_elements: int,
):
    """
    Calculate the scaffold value
    """
    no_goal_above = np.logical_not(goal_occupancy_grid[:, :, support_height + 1] & current_occupancy_grid[:, :, support_height + 1])
    current_blocks = current_occupancy_grid[:, :, support_height]
    goal_blocks = goal_occupancy_grid[:, :, support_height]

    useful_blue = support[1] & (current_blocks | goal_blocks) & no_goal_above
    if np.any(useful_blue):
        return True, False, False

    useful_red = support[0] & (current_blocks | goal_blocks) & no_goal_above
    if np.sum(useful_red) >= max(1, goal_height - support_height - (number_of_group_elements == 1)):
        return True, False, False

    valid = np.any((support[0] | support[1]) & no_goal_above)
    return valid, True, (support[1] & no_goal_above)


def cal_goal_val(world_size: Tuple[int, int, int], group2support: List[NDArray], useful_support: List[NDArray], scaffold_val: NDArray) -> float:
    """
    compute an underestimate of the number of scaffolding blocks needed to be placed
    """
    scaffold_v = np.sum(scaffold_val, axis=0)
    goal_v = 0.0
    for height in range(world_size[2] - 1):
        supports_at_height = group2support[height].sum(axis=2)
        num_useful_at_height = useful_support[height]
        num_groups = num_useful_at_height.shape[0]
        for i in range(num_groups):
            val = (scaffold_v * supports_at_height[i]).max(axis=(1, 2))
            val = np.divide(num_useful_at_height[i], val, out=np.zeros_like(val, dtype=np.float32), where=val > 0)
            goal_v += float(val.sum())
    return goal_v


def estimate_missing_scaffolding(current_occupancy_grid: NDArray, goal_occupancy_grid: NDArray):
    groups = find_groups(current_occupancy_grid.shape, goal_occupancy_grid, current_occupancy_grid)
    group2support, sizes = find_group_support(current_occupancy_grid.shape, groups)
    scaffold_val, useful_support = cast_scaffold_value(current_occupancy_grid.shape, goal_occupancy_grid, group2support, sizes, current_occupancy_grid)
    return cal_goal_val(current_occupancy_grid.shape, group2support, useful_support, scaffold_val)
