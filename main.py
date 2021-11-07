import seaborn as sns
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class Astar:

    def __init__(self, matrix):
        self.mat = self.prepare_matrix(matrix)

    class Node:
        def __init__(self, x, y, weight=0):
            self.x = x
            self.y = y
            self.weight = weight
            self.cost=0
            self.parent = None

        # def __repr__(self):
        #     return str(((self.x, self.y), self.manhattan))

    def print(self):
        for y in self.mat:
            print(y)

    def prepare_matrix(self, mat):
        matrix_for_astar = []
        for y, line in enumerate(mat):
            tmp_line = []
            for x, weight in enumerate(line):
                tmp_line.append(self.Node(x, y, weight=weight))
            matrix_for_astar.append(tmp_line)
        return matrix_for_astar

    def equal(self, current, end):
        return current.x == end.x and current.y == end.y

    def manhattan(self, current, other):
        return abs(current.x - other.x) + abs(current.y - other.y)

    def neighbours(self, matrix, current):
        neighbours_list = []
        if current.x - 1 >= 0 and matrix[current.y][current.x - 1].weight is not None:
            neighbours_list.append(matrix[current.y][current.x - 1])
        if current.y - 1 >= 0 and matrix[current.y - 1][current.x].weight is not None:
            neighbours_list.append(matrix[current.y - 1][current.x])
        if current.y + 1 < len(matrix) and matrix[current.y + 1][current.x].weight is not None:
            neighbours_list.append(matrix[current.y + 1][current.x])
        if current.x + 1 < len(matrix[0]) and matrix[current.y][current.x + 1].weight is not None:
            neighbours_list.append(matrix[current.y][current.x + 1])
        return neighbours_list




    def reconstruct_path(self, end):
        node_tmp = end
        path = []
        while (node_tmp):
            path.append([node_tmp.x, node_tmp.y])
            node_tmp = node_tmp.parent
        return list(reversed(path))

    def run(self, point_start, point_end):
        matrix = self.mat

        start = self.Node(point_start[0], point_start[1])
        end = self.Node(point_end[0], point_end[1])
        closed_list = []
        open_list = [start]

        def sort_key(node):
            return node.cost

        while open_list:

            open_list.sort(key=sort_key)
            current_node =  open_list.pop(0)

            for node in open_list:
                if node.cost < current_node.cost and node not in closed_list:
                    current_node = node

            if self.equal(current_node, end):
                print(closed_list)
                return self.reconstruct_path(current_node)

            for node in open_list:
                if self.equal(current_node, node):
                    open_list.remove(node)
                    break

            closed_list.append(current_node)

            for neighbour in self.neighbours(matrix, current_node):
                if neighbour in closed_list:
                    continue
                if (neighbour.cost < current_node.cost) or (neighbour not in open_list):
                    neighbour.parent = current_node
                    neighbour.cost = self.calculate_parents_weight(neighbour) + self.manhattan(neighbour, end)

                if neighbour not in open_list:
                    open_list.append(neighbour)

        return None

    def calculate_parents_weight(self, node):
        weight_sum = node.weight
        while node.parent is not None:
            weight_sum += node.parent.weight
            node= node.parent
        return weight_sum

    def map_projection(self, map, result):
        for i, row in enumerate(map):
            for j, item in enumerate(row):
                if item is None:
                    map[i][j]=np.NAN

        sns.set_theme()
        ax =sns.heatmap(map2,linewidths=.5,annot=True, cbar=False)
        result_df = pd.DataFrame(result, columns=['x', 'y'])
        print(result_df)
        plt.plot(result_df.x + 0.5, result_df.y + 0.5, linewidth=10)
        plt.show()

    def maps(self, map2, map):
        for n in range(len(map)):
            for k in range(len(map[0])):
                if map2[n][k] is not None:
                    map2[n][k] += 10
                else:
                    map2[n][k] = None

        for n in range(len(map)):
            for k in range(len(map[0])):
                if map2[n][k] is not None and map[n][k] is not None:
                    map2[n][k] += map[n][k]
                else:
                    map2[n][k] = None


if __name__== "__main__":
    map2 = [
        [1, 1, 1, 3, 2, 3, 3, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3],
        [2, 3, 3, 1, 2, 3, 3, None, None, None, 3, 4, 4, 4, 1, 1, 1],
        [3, 3, 3, None, 1, 1, 1, 1, 1, 1, 5, 1, 1, None, 1, 5, 5],
        [2, None, None, None, 1, 2, 2, None, None, None, 3, 4, None, 6, 7, 6, 6],
        [2, 2, 2, 3, 1, 2, 2, 3, 3, 3, 3, 3, 5, 6, 7, 6, 6],
        [4, 2, 2, None, 1, 2, 2, 3, 3, 3, 3, 4, 5, 7, 8, 9, 8],
        [None, 2, 2, None, 1, 3, None, 3, 3, 3, 3, 4, 5, 7, 9, 10, 8],
        [3, 3, 3, None, 1, 3, None, 3, 3, 3, 3, 4, None, 6, 9, 10, 8],
        [4, None, 4, None, 1, 5, 5, 3, 3, 3, 3, 4, 5, 7, 8, 9, 8],
        [4, 4, 6, 6, 1, 3, 3, 3, 5, 5, 5, 5, None, 6, 8, 8, 6],
        [None, 4, None, None, 1, 1, 1, 1, 1, 5, 5, 5, 5, 2, 1, 6, 6],
        [3, 4, None, 4, 3, 3, 3, None, 1, 5, 5, 5, 5, 3, None, 4, 4],
        [4, 4, 5, 4, 4, 3, None, None, 1, 5, 5, 5, 5, 3, 1, 4, 4],
        [4, 6, 6, 6, 3, 3, 3, 2, 1, 1, 4, None, 1, 1, 1, 3, 3],
        [4, 6, 8, 8, 6, 3, 3, 2, 2, 2, 4, 4, 2, 2, 1, 3, 4],
    ]
    map = [
         [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
         [   0,    0,    0,    0,    0,   -8,    0, None, None, None,    0,    0,    0,    0,    0,    0,    0],
         [   0,    0,    0, None,   -8,   -9,   -8,    0,    0,    0,    0,    0,    0, None, None, None,    0],
         [   0, None, None, None,   -9,  -10,   -9, None, None, None, None, None, None,   -8,   -9,   -8,    0],
         [   0,    0,    0, None,   -8,   -9,   -8,    0,    0,    0,    0,    0,   -8,   -9,  -10,   -9,   -8],
         [   0,    0,    0, None,    0,   -8,    0,    0,    0,    0,    0,    0,    0,   -8,   -9,   -8,    0],
         [None,    0,    0, None,    0,    0, None,    0,    0,    8,    0,    0,    0,    0,   -8,    0,    0],
         [   0,    0,    0, None,    0,    0, None,    0,    8,   9,    8,    0, None,    0,    0,    0,    0],
         [   0, None,    0, None,    0,    0,    0,    8,    9,   10,    9,    8,    0,    0,    0,    0,    0],
         [   0,    0,    0, None,    0,    0,    0,    8,    8,    9,    8,    0, None,    0,    0,    0,    0],
         [None,    0, None, None,    0,    0,    0,    0,    0,    8,    0,    0,    0,    0,    0,    0,    0],
         [   0,    0, None,    0,    0,    0,    0, None,    0,    0,    0,    0,    0,    0, None,    0,    0],
         [   0,    0,    8,    9,    8,    0, None, None,    0,    0,    0,    0,    0,    0,    0,    0,    0],
         [   0,    8,    9,   10,    9,    8,    0,    0,    0,    0,    0, None,    0,    0,    0,    0,    0],
         [   0,    0,    8,    9,    8,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0]
    ]

    #Astar.maps(map2, map2, map)
    astar = Astar(map2)
    result = astar.run([0, 0], [16, 9])
    print(result)
    astar.map_projection(map2,result)




