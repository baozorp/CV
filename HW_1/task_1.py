import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    # Предполагаем, что вход находится в первой строке, а выход - в последней
    start = (0, np.where((image[0] == [255, 255, 255]).all(axis=1))[0][0])
    end_row = image.shape[0] - 1

    # Инициализация очереди и множества посещенных узлов
    queue = deque([start])
    visited = set()
    visited.add(start)
    
    # Направления движения: вверх, вниз, влево, вправо
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Хранение пути
    parent = {start: None}

    while queue:
        current = queue.popleft()

        # Проверяем, достигли ли мы конца
        if current[0] == end_row:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            x, y = zip(*path)
            return (np.array(x), np.array(y))

        # Обработка соседних узлов
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            # Проверяем, что сосед находится в пределах изображения и проходим
            if (0 <= neighbor[0] < image.shape[0] and
                0 <= neighbor[1] < image.shape[1] and
                (neighbor not in visited) and
                (image[neighbor[0], neighbor[1]] == [255, 255, 255]).all()):
                
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current

    return None  # Если пути не найдено
