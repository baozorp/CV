import numpy as np
import torch

def conv_nested(image_data, kernel_matrix):
    """ Проводит свертку изображения с использованием вложенного цикла. """
    img_height, img_width = image_data.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))

    padded_image = np.pad(image_data, kernel_height // 2)

    for row in range(img_height):
        for col in range(img_width):
            for k_row in range(kernel_height):
                for k_col in range(kernel_width):
                    output[row, col] += padded_image[row + k_row, col + k_col] * kernel_matrix[kernel_height - 1 - k_row, kernel_width - 1 - k_col]

    return output

def zero_pad(image_data, pad_height, pad_width):
    """ Применяет нулевое заполнение к изображению. """
    height, width = image_data.shape
    padded_output = np.zeros_like(image_data)

    if pad_width > pad_height:
        padded_output = np.pad(image_data, pad_height)
        h, w = padded_output.shape
        surplus = pad_width - pad_height
        padded_output = np.insert(padded_output, [0]*surplus, 0, axis=1)
        padded_output = np.insert(padded_output, [-1]*surplus, 0, axis=1)
    else:
        padded_output = np.pad(image_data, pad_width)
        h, w = padded_output.shape
        surplus = pad_height - pad_width
        padded_output = np.insert(padded_output, [0]*surplus, 0, axis=0)
        padded_output = np.insert(padded_output, [-1]*surplus, 0, axis=0)

    return padded_output

def apply_zero_padding(image_data, pad_height, pad_width):
    """ Применяет нулевое заполнение к изображению. """
    height, width = image_data.shape
    padded_output = np.zeros_like(image_data)

    if pad_width > pad_height:
        padded_output = np.pad(image_data, pad_height)
        h, w = padded_output.shape
        surplus = pad_width - pad_height
        padded_output = np.insert(padded_output, [0]*surplus, 0, axis=1)
        padded_output = np.insert(padded_output, [-1]*surplus, 0, axis=1)
    else:
        padded_output = np.pad(image_data, pad_width)
        h, w = padded_output.shape
        surplus = pad_height - pad_width
        padded_output = np.insert(padded_output, [0]*surplus, 0, axis=0)
        padded_output = np.insert(padded_output, [-1]*surplus, 0, axis=0)

    return padded_output

def conv_fast(image_data, kernel_matrix):
    """ Проводит быструю свертку изображения с использованием свертки. """
    img_height, img_width = image_data.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))

    padded_image = apply_zero_padding(image_data, kernel_height // 2, kernel_width // 2)
    kernel_matrix = np.flip(kernel_matrix)

    for row in range(img_height):
        for col in range(img_width):
            output[row, col] = np.sum(padded_image[row:row + kernel_height, col:col + kernel_width] * kernel_matrix)

    return output

def conv_faster(image_data, kernel_matrix):
    """ Проводит свертку изображения с использованием PyTorch для ускорения. """
    img_height, img_width = image_data.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output = np.zeros((img_height, img_width))

    kernel_matrix = np.flip(kernel_matrix)
    torch_kernel = torch.tensor(kernel_matrix.copy(), dtype=torch.float32).reshape(1, 1, kernel_height, kernel_width)

    conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_height, kernel_width), padding='same', bias=False)
    conv_layer.weight = torch.nn.Parameter(torch_kernel)

    torch_image = torch.tensor(image_data, dtype=torch.float32).reshape(1, 1, img_height, img_width)
    output = conv_layer(torch_image).detach().numpy().reshape((img_height, img_width))

    return output

def cross_correlation(image_a, image_b):
    """ Выполняет корреляцию между двумя изображениями. """
    img_b_float = image_b.astype(np.float64)
    img_a_float = image_a.astype(np.float64)

    img_height, img_width = img_a_float.shape
    kernel_height, kernel_width = img_b_float.shape
    output = np.zeros((img_height, img_width))
    padded_image = apply_zero_padding(img_a_float, kernel_height // 2, kernel_width // 2)
    sum_sq_img_b = np.sum(img_b_float ** 2)

    for row in range(img_height):
        for col in range(img_width):
            img_slice = padded_image[row:row + kernel_height, col:col + kernel_width]
            coefficient = np.sqrt(sum_sq_img_b * np.sum(img_slice ** 2))
            output[row, col] = np.sum(img_slice * img_b_float) / coefficient

    return output

def zero_mean_cross_correlation(image_a, image_b):
    """ Выполняет корреляцию с нулевым средним изображением. """
    temp_kernel = image_b - np.mean(image_b)
    return cross_correlation(image_a, temp_kernel)

def normalized_cross_correlation(image_a, image_b):
    """ Выполняет нормализованную корреляцию между двумя изображениями. """
    img_b_float = image_b.astype(np.float64)
    img_a_float = image_a.astype(np.float64)

    img_height, img_width = img_a_float.shape
    kernel_height, kernel_width = img_b_float.shape
    output = np.zeros((img_height, img_width))
    padded_image = apply_zero_padding(img_a_float, kernel_height // 2, kernel_width // 2)

    std_dev = np.std(img_b_float)
    mean_val = np.mean(img_b_float)
    norm_img_b = (img_b_float - mean_val) / std_dev
    sum_sq_img_b = np.sum(img_b_float ** 2)

    for row in range(img_height):
        for col in range(img_width):
            img_slice = padded_image[row:row + kernel_height, col:col + kernel_width]
            coefficient = np.sqrt(sum_sq_img_b * np.sum(img_slice ** 2))
            output[row, col] = np.sum(((img_slice - np.mean(img_slice)) / np.std(img_slice)) * norm_img_b) / coefficient

    return output
