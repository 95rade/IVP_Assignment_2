import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as read
import cv2

MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255


def load_image(name, usecv=True, type=None):
    if usecv:
        if type is not None:
            return cv2.imread(name, type)
        else:
            return cv2.imread(name)
    else:
        return read.imread(name)


def plot_image(image, usecv=True):
    if usecv:
        cv2.imshow('image', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.show()


def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.
    Parameters
    ----------
    figures : list of (name, image)
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, figure in enumerate(figures):
        name, image = figure
        axeslist.ravel()[ind].imshow(image, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(name)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optional
    plt.show()


def generate_histogram(image):
    h, w = image.shape
    histogram = np.bincount(np.resize(image, (1, h * w))[0])

    if histogram.size != 256:
        histogram = np.append(histogram, np.zeros(256 - histogram.size, dtype=int))

    return histogram


def plot_histogram(histo):
    plt.figure()
    plt.bar(list(range(0, histo.size)), histo, align='center')
    plt.show()


def linear_transformation_to_pixel_value_range(data, min, max):
    data = np.array(data, dtype=np.float64)
    new_data = 255 + (255 * (data - max) / (max - min))
    return np.rint(new_data)


# def log_transformation(img, c):
#     # Image pixels can overflow and wrap around
#     image = np.array(img, dtype=np.float64) / MAX_PIXEL_VALUE
#     new_image_data = np.log(1 + image) * c
#     # linear_transform = new_image_data
#     linear_transform = linear_transformation_to_pixel_value_range(new_image_data, 0, np.log(2)).astype(np.uint8)
#     # plot_histogram(generate_histogram(img))
#     # plot_histogram(generate_histogram(linear_transform))
#     return (np.rint(linear_transform)).astype(np.uint8)


def log_transformation(img, c):
    # Image pixels can overflow and wrap around
    image = np.array(img, dtype=np.float64) / MAX_PIXEL_VALUE
    new_image_data = np.log(1 + image) * c
    linear_transform = new_image_data
    linear_transform = linear_transformation_to_pixel_value_range(new_image_data, 0, np.log(2) * c).astype(np.uint8)
    # plot_histogram(generate_histogram(img))
    # plot_histogram(generate_histogram(linear_transform))
    return (np.rint(linear_transform)).astype(np.uint8)


def power_law_transformation(image, c, y):  # y = gamma
    data = np.array(image, dtype=np.float64) / MAX_PIXEL_VALUE
    transformed_image = c * np.power(data, y)
    linear_transform = linear_transformation_to_pixel_value_range(transformed_image, 0, c)
    return linear_transform.astype(np.uint8)


def image_transformation():
    """
    log transformation
    """
    image = load_image('images/spine.tif', type=cv2.IMREAD_GRAYSCALE)
    log_transform = log_transformation(image.copy(), 300)
    plot_image(log_transform.copy())

    """
    power law transformation
    """
    power_law_transform = power_law_transformation(image.copy(), 1, 0.4)
    # plot_image(image.copy())
    plot_image(power_law_transform.copy())


def histogram_generation():
    bright = load_image('images/beads_bright.tif', type=cv2.IMREAD_GRAYSCALE)
    normal = load_image('images/beads_normal.tif', type=cv2.IMREAD_GRAYSCALE)
    dark = load_image('images/beads_dark.tif', type=cv2.IMREAD_GRAYSCALE)

    plot_histogram(generate_histogram(bright))
    plot_histogram(generate_histogram(normal))
    plot_histogram(generate_histogram(dark))


def equalise_histogram(image):
    histo = generate_histogram(image)
    new_image = np.zeros(image.shape)

    height, width = image.shape
    total_pixels = height * width

    freq_sum = np.sum(histo)

    for i, freq in enumerate(reversed(histo)):
        intensity = 255 - i
        new_intensity = round(255 * freq_sum / total_pixels)

        temp = image.copy()
        np.place(temp, temp != intensity, 0)
        np.place(temp, temp == intensity, new_intensity)

        new_image += temp

        freq_sum -= freq

    new_image = np.array(new_image, dtype=np.uint8)
    new_histo = generate_histogram(new_image)
    plot_histogram(new_histo)

    return new_image


def histogram_equalisation():
    bright = load_image('images/beads_bright.tif', type=cv2.IMREAD_GRAYSCALE)
    plot_image(equalise_histogram(bright.copy()))

    normal = load_image('images/beads_normal.tif', type=cv2.IMREAD_GRAYSCALE)
    plot_image(equalise_histogram(normal.copy()))

    dark = load_image('images/beads_dark.tif', type=cv2.IMREAD_GRAYSCALE)
    plot_image(equalise_histogram(dark.copy()))


def apply_filter(image, img_filter, horizontally=True):
    filter_size = img_filter.shape
    filter_size = filter_size[0]

    pad = filter_size // 2
    height, width = image.shape

    new_image = np.zeros((height, width), dtype=np.uint8)

    if horizontally:
        for h in range(pad, height - pad):
            for w in range(pad, width - pad):
                new_image[h, w] = np.clip(np.rint(
                    np.sum(np.multiply(image[h - pad: h + pad + 1, w - pad: w + pad + 1], img_filter))),
                    MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)

                aa = np.sum(np.multiply(image[h - pad: h + pad + 1, w - pad: w + pad + 1], img_filter))
    else:
        for w in range(pad, width - pad):
            for h in range(pad, height - pad):
                new_image[h, w] = np.clip(np.rint(
                    np.sum(np.multiply(image[h - pad: h + pad + 1, w - pad: w + pad + 1], img_filter))),
                    MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)

    return new_image


def simple_average(size):
    image = load_image('images/tapsee-pannu.jpg', type=cv2.IMREAD_GRAYSCALE)
    image_filter = np.ones((size, size), np.float32) / (size ** 2)

    figures = []

    figures.append(('Orignal', image))
    figures.append((str(image_filter.shape) + ' Average', apply_filter(image.copy(), image_filter)))

    plot_figures(figures, 1, 2)


def median_noise_filter(image, window_size):
    pad = window_size // 2
    height, width = image.shape

    new_image = np.zeros((height - (2 * pad), width - (2 * pad)), dtype=np.uint8)

    for h in range(pad, height - pad):
        print("Processing height: " + str(h) + "/" + str(height))
        for w in range(pad, width - pad):
            new_image[h - pad, w - pad] = np.ma.median(image[h - pad: h + pad + 1, w - pad: w + pad + 1])

    return new_image


def multiply_images(img1, img2):
    img1 = np.array(img1, dtype=int)
    img2 = np.array(img2, dtype=int)

    multi = np.multiply(img1, img2)

    result = linear_transformation_to_pixel_value_range(multi, 0, MAX_PIXEL_VALUE ** 2)
    return result.astype(np.uint8)


def add_images(img1, img2):
    img1 = np.array(img1, dtype=int)
    img2 = np.array(img2, dtype=int)

    result = linear_transformation_to_pixel_value_range(img1 + img2, 0,  2 * MAX_PIXEL_VALUE)
    return result.astype(np.uint8)


def Q5():
    image = load_image('images/skeleton.tif', type=cv2.IMREAD_GRAYSCALE)

    l_filter = np.zeros((3, 3), dtype=np.int64)
    l_filter[0, 1] = 1
    l_filter[1, 0] = 1
    l_filter[1, 2] = 1
    l_filter[2, 1] = 1
    l_filter[1, 1] = -4

    s_filter_v = np.zeros((3, 3), dtype=np.int64)
    s_filter_v[0, 0] = -1
    s_filter_v[0, 1] = -2
    s_filter_v[0, 2] = -1
    s_filter_v[2, 0] = 1
    s_filter_v[2, 1] = 2
    s_filter_v[2, 2] = 1

    s_filter_h = np.zeros((3, 3), dtype=np.int64)
    s_filter_h[0, 0] = -1
    s_filter_h[1, 0] = -2
    s_filter_h[2, 0] = -1
    s_filter_h[0, 2] = 1
    s_filter_h[1, 2] = 2
    s_filter_h[2, 2] = 1

    image_laplacian_sharpened = image + apply_filter(image.copy(), l_filter)
    image_sobel = apply_filter(image.copy(), s_filter_h) + apply_filter(image.copy(), s_filter_v, False)
    image_sobel_5x5_average = apply_filter(image_sobel, np.ones((5, 5)) / 25)

    product_lap_sobel_5x5 = multiply_images(image_laplacian_sharpened, image_sobel_5x5_average)
    orig_plus_prod = add_images(image.copy(), product_lap_sobel_5x5.copy())

    power_law = power_law_transformation(orig_plus_prod, 1, 0.6)

    plot_image(image, False)
    plot_image(image_laplacian_sharpened, False)
    plot_image(image_sobel, False)
    plot_image(image_sobel_5x5_average, False)
    plot_image(linear_transformation_to_pixel_value_range(product_lap_sobel_5x5, 0, np.max(product_lap_sobel_5x5).copy()), False)
    plot_image(orig_plus_prod, False)
    plot_image(power_law, False)


def Q6():
    image = load_image('images/tapsee-pannu.jpg', type=cv2.IMREAD_GRAYSCALE)
    # template = load_image('images/tapsee-pannu-eye.jpg', type=cv2.IMREAD_GRAYSCALE)

    i_height, i_width = image.shape
    tx, ty = (160, 230)
    t_height, t_width = (80, 100)
    template = image[tx: tx + t_height, ty: ty + t_width]

    t_height, t_width = template.shape

    for h in range(0, i_height - t_height):
        print("processing height: " + str(h) + " / " + str(i_height))
        found = False
        for w in range(0, i_width - t_width):
            if np.allclose(image[h: h + t_height, w: w + t_width], template):
                image[h: h + t_height, w: w + 1] = 255
                image[h: h + t_height, w + t_width - 1: w + t_width] = 255
                image[h: h + 1, w: w + t_width] = 255
                image[h + t_height - 1: h + t_height, w + t_width - 1: w + t_width] = 255
                plot_image(image)

                found = True
                break
        if found:
            break


def main():
    # image_transformation()            # Q1
    # histogram_generation()            # Q2
    # histogram_equalisation()          # Q3
    # simple_average(3)                  # Q4
    # simple_average(11)                  # Q4
    # simple_average(21)                  # Q4
    # Q5()
    Q6()


if __name__ == "__main__":
    main()
