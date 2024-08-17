import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

# 随机生成32x32的OGM地图
from collections import deque
import time

def generate_clustered_bitmap_bfs(size=64, proportion=1/6):
    # Initialize a bitmap with zeros
    bitmap = np.zeros((size, size), dtype=int)
    num_ones = int(size * size * np.random.uniform(0.08, 0.25))
    
    # Choose a random starting point
    start_row = random.randint(25, 40)
    start_col = random.randint(25, 40)
    queue = deque([(start_row, start_col)])
    visited = set(queue)
    ones_added = 0

    # 生成一组4个，0-7的随机整数
    random_int = random.randint(0, 7)
    random_int2 = random.randint(0, 6)
    random_int3 = random.randint(0, 5)

    while ones_added < num_ones and queue:
        row, col = queue.popleft()
        bitmap[row, col] = 1
        ones_added += 1
        
        # Get neighboring cells
        neighbors = [
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1),
            (row - 1, col - 1), (row - 1, col + 1),
            (row + 1, col - 1), (row + 1, col + 1),
        ]
        neighbors.pop(random_int)
        neighbors.pop(random_int2)
        neighbors.pop(random_int3)
        # Shuffle neighbors to randomize the spread
        #random.shuffle(neighbors)
        
        for r, c in neighbors:
            if 0 <= r < size and 0 <= c < size and (r, c) not in visited:
                if np.random.uniform() < 0.8:
                    queue.append((r, c))
                    visited.add((r, c))
    
    return bitmap

# A*算法实现
def astar(grid, start, end):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if  -5 < current[0] - end[0] < 5 and -5 < current[1] - end[1] < 5:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (-2, 2), (2, -2), (2, 2)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 < neighbor[0] < len(grid)-1 and 0 < neighbor[1] < len(grid[0])-1 and np.sum(grid[neighbor[0]-1:neighbor[0]+2,neighbor[1]-1:neighbor[1]+2]) <= 0.5:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None


def dfs(grid, start, end):

    stack = [(start, [start])]
    visited = np.zeros_like(grid)
    seq = [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (-2, 2), (2, -2), (2, 2)]
    rel = [start[0] - end[0], start[1] - end[1]]
    seq = sorted(seq, key=lambda x: rel[0] * x[0] + rel[1] * x[1])
    while stack:
        (current, path) = stack.pop()
        if visited[current] == 1:
            continue
        visited[current] = 1

        
        if -5 < current[0] - end[0] < 5 and -5 < current[1] - end[1] < 5:
            return path

        def search(current,seq):
            for dx, dy in seq:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                    if visited[neighbor] == 0:
                        stack.append((neighbor, path + [neighbor]))
                    
    return None

class Goal:
    def __init__(self, grid = np.zeros((64,64))):
        self.grid = grid
        self.start = None
        self.end = None
        self.visited = None
        self.seq = [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (-2, 2), (2, -2), (2, 2)]
        self.rel = None
        self.path = None
    def search(self,current):
        if -5 < current[0] - self.end[0] < 5 and -5 < current[1] - self.end[1] < 5:
            return True   
        self.rel = [self.end[0] - current[0] , self.end[1] - current[1]]
        self.seq = sorted(self.seq, key=lambda x: self.rel[0] * x[0] + self.rel[1] * x[1], reverse=True)
        for dx, dy in self.seq:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(self.grid) and 0 <= neighbor[1] < len(self.grid[0]) and np.sum(self.grid[neighbor[0]-1:neighbor[0]+2,neighbor[1]-1:neighbor[1]+2]) <= 0.5:
                if self.visited[neighbor] == 0:
                    self.path.append(neighbor)
                    self.visited[neighbor] = 1
                    if self.search(neighbor):
                        return True
                    self.visited[neighbor] = 0
                    self.path.pop()
        return False                    

    def get_goal(self,start,end):
        self.start = start
        self.end = end
        self.visited = np.zeros_like(self.grid)
        self.path = [start]
        self.visited[start] = 1
        if self.search(self.start):
            return self.path
        return None
        
    # 判断起点和终点之间是否有障碍物
    def is_valid(self, start, end):
        # if self.grid[start[0]][start[1]] == 1 or self.grid[end[0]][end[1]] == 1:
        #     return False
        
        # # Get all the integer points between start and end
        # points = line(start[0], start[1], end[0], end[1])
        
        # # Check if any of the points have obstacles
        # for point in points:
        #     if self.grid[point[0]][point[1]] == 1:
        #     return False
        
        return True

# 可视化地图和路径
def visualize_path(grid, path, start, end):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='Greys', origin='upper')
    plt.scatter(start[1], start[0], c='blue', marker='o', label='Start')
    plt.scatter(end[1], end[0], c='red', marker='x', label='End')
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], c='yellow', linewidth=2, label='Path')
    plt.legend()
    plt.grid(True)
    plt.show()

# 生成地图
size = (64,64)
grid = generate_clustered_bitmap_bfs(64)

# 随机选择起点和终点，确保它们在空地上
def random_point_on_grid(grid):
    while True:
        point = (random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[1] - 1))
        if grid[point] == 0:
            return point


if __name__ == '__main__':
    # 寻找路径
    # Repeat A* 1000 times and measure average runtime
    total_time = 0


    # print("Start:", start)
    # print("End:", end)
    Planner = Goal(grid)

    for _ in range(10000):
        
        start = random_point_on_grid(grid)
        end = random_point_on_grid(grid)
        while start[0] - end[0] + start[1] - end[1] < 50:
            start = random_point_on_grid(grid )
            end = random_point_on_grid(grid)
        
        start_time = time.time()
        path = Planner.get_goal(start,end)#dfs(grid, start, end)
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / 10000
    print("Average runtime:", average_time)

    # 可视化
    visualize_path(grid, path, start, end)
