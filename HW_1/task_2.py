import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер полосы, на которой нет препятствия в конце пути.

    :param image: исходное изображение
    :return: номер полосы, на которой нет препятствия в конце дороги
    """
    # Преобразуем изображение в цветовую модель HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Определяем диапазоны цветов для красных препятствий
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    
    # Создаем маску для красного цвета
    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = mask_red_1 | mask_red_2
    
    # Находим контуры красных препятствий
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Определяем размеры изображения и количество полос
    height, width, _ = image.shape
    num_lanes = 5  
    lane_width = width // num_lanes
    
    # Массив для отслеживания полос с препятствиями
    lanes_with_obstacles = [False] * num_lanes
    
    # Обрабатываем контуры и отмечаем полосы с препятствиями
    for cnt in contours_red:
        x, y, w, h = cv2.boundingRect(cnt)
        lane_index = x // lane_width   
        if lane_index < num_lanes:  
            lanes_with_obstacles[lane_index] = True
    
    # Проверяем, есть ли свободные полосы
    for i in range(num_lanes):
        if not lanes_with_obstacles[i]:
            return i  
    
    return -1  # Если все полосы заняты, возвращаем -1


