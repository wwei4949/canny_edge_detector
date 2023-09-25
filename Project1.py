import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt

### Load, convert to grayscale, plot, and resave an image
I = np.array(open('Iribe.jpg').convert('L')) / 255

plt.imshow(I, cmap='gray')
plt.axis('off')
plt.title("Original Image")
plt.show()


### Part 1
# Forms a 3sigma × 3sigma Gaussian kernel with standard deviation sigma. Sigma is a positive integer.
def gausskernel(sigma):
    size = 3 * sigma
    # Create a 2D grid of x and y values
    [x, y] = np.meshgrid(np.arange(-1 * (size - 1) / 2, (size - 1) / 2 + 1),
                         np.arange(-1 * (size - 1) / 2, (size - 1) / 2 + 1))
    # Calculate the Gaussian kernel
    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    # Normalize the kernel
    h = h / np.sum(h)
    return h


# Convolves an image with a given filter.
def myfilter(I, h):
    (I_height, I_width) = I.shape
    (h_height, h_width) = h.shape
    pad_left = int((h_width - 1) / 2)
    pad_right = int((h_width - 1) / 2)
    pad_top = int((h_height - 1) / 2)
    pad_bottom = int((h_height - 1) / 2)

    # Appropriately pad I (half of the kernel size on each side with 0s)
    I_padded = np.zeros((I_height + pad_bottom + pad_top, I_width + pad_left + pad_right))
    (I_padded_height, I_padded_width) = I_padded.shape
    I_padded[pad_top:I_padded_height - pad_bottom, pad_left:I_padded_width - pad_right] = I

    # Convolve I with h
    I_ft = np.fft.fft2(I, s=(I_padded_height, I_padded_width))
    h_ft = np.fft.fft2(h, s=(I_padded_height, I_padded_width))
    I_ft = h_ft * I_ft
    I_filtered = np.fft.ifft2(I_ft)
    I_filtered = np.real(I_filtered)
    I_filtered = I_filtered[pad_top:I_padded_height - pad_bottom, pad_left:I_padded_width - pad_right]
    return I_filtered


# Testing part 1
I_filtered = myfilter(I, gausskernel(3))
plt.imshow(I_filtered, cmap='gray')
plt.axis('off')
plt.title("Filtered Image Using a 9X9 Gaussian Kernel")
plt.show()
I_filtered = myfilter(I, gausskernel(5))
plt.imshow(I_filtered, cmap='gray')
plt.axis('off')
plt.title("Filtered Image Using a 15X15 Gaussian Kernel")
plt.show()
I_filtered = myfilter(I, gausskernel(10))
plt.imshow(I_filtered, cmap='gray')
plt.axis('off')
plt.title("Filtered Image Using a 30X30 Gaussian Kernel")
plt.show()
I_gauss = gausskernel(10)
plt.imshow(I_gauss, cmap='gray')
plt.axis('off')
plt.title("Gaussian Kernel with sigma = 10")
plt.show()

# 3.3 Filters
h1 = np.array([[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 2, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]])
h2 = np.array([[-1, 3, -1]])
h3 = np.array([[-1], [3], [-1]])
I_filtered = myfilter(I, h1)
plt.imshow(I_filtered, cmap='gray')
plt.axis('off')
plt.title("Filtered Image Using h1 Filter")
plt.show()
I_filtered = myfilter(I, h2)
plt.imshow(I_filtered, cmap='gray')
plt.axis('off')
plt.title("Filtered Image Using h2 Filter")
plt.show()
I_filtered = myfilter(I, h3)
plt.imshow(I_filtered, cmap='gray')
plt.axis('off')
plt.title("Filtered Image Using h3 Filter")
plt.show()

### Part 2
from scipy.ndimage import label


# Returns True when the edge at (x,y) should be thinned and False otherwise.
def check_thin(x, y, angles, magnitudes):
    if angles[x][y] == 0:
        if magnitudes[x][y] < magnitudes[x][y - 1] or magnitudes[x][y] < magnitudes[x][y + 1]:
            return True
    elif angles[x][y] == 45:
        if magnitudes[x][y] < magnitudes[x - 1][y + 1] or magnitudes[x][y] < magnitudes[x + 1][y - 1]:
            return True
    elif angles[x][y] == 90:
        if magnitudes[x][y] < magnitudes[x - 1][y] or magnitudes[x][y] < magnitudes[x + 1][y]:
            return True
    elif angles[x][y] == 135:
        if magnitudes[x][y] < magnitudes[x - 1][y - 1] or magnitudes[x][y] < magnitudes[x + 1][y + 1]:
            return True
    return False


# helper function to calculate theta prime
def convert_theta_prime(theta):
    theta_p = np.copy(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            while theta_p[i][j] >= 180:
                theta_p[i][j] -= 180
            if theta_p[i][j] < 0:
                theta_p[i][j] *= -1
            if 0 <= theta_p[i][j] < 22.5:
                theta_p[i][j] = 0
            elif 22.5 <= theta_p[i][j] < 67.5:
                theta_p[i][j] = 45
            elif 67.5 <= theta_p[i][j] < 112.5:
                theta_p[i][j] = 90
            elif 112.5 <= theta_p[i][j] < 157.5:
                theta_p[i][j] = 135
            elif 157.5 <= theta_p[i][j] < 180:
                theta_p[i][j] = 0
    return theta_p


# check a component, if any pixel in that component has a magnitude greater than t_high, set all to be an edge
def check_component(x, y, labeled, magnitudes, t_high):
    has_high = False
    i = x
    j = y
    # check left, right, up, down, and diagonals
    while j > 0 and labeled[i][j] == labeled[i][j - 1]:
        if magnitudes[i][j - 1] > t_high:
            has_high = True
            break
        j -= 1
    if not has_high:
        i = x
        j = y
        while j < magnitudes.shape[1] - 1 and labeled[i][j] == labeled[i][j + 1]:
            if magnitudes[i][j + 1] > t_high:
                has_high = True
                break
            j += 1
    if not has_high:
        i = x
        j = y
        while i > 0 and labeled[i][j] == labeled[i - 1][j]:
            if magnitudes[i - 1][j] > t_high:
                has_high = True
                break
            i -= 1
    if not has_high:
        i = x
        j = y
        while i < magnitudes.shape[0] - 1 and labeled[i][j] == labeled[i + 1][j]:
            if magnitudes[i + 1][j] > t_high:
                has_high = True
                break
            i += 1
    if not has_high:
        i = x
        j = y
        while i > 0 and j > 0 and labeled[i][j] == labeled[i - 1][j - 1]:
            if magnitudes[i - 1][j - 1] > t_high:
                has_high = True
                break
            i -= 1
            j -= 1
    if not has_high:
        i = x
        j = y
        while i > 0 and j < magnitudes.shape[1] - 1 and labeled[i][j] == labeled[i - 1][j + 1]:
            if magnitudes[i - 1][j + 1] > t_high:
                has_high = True
                break
            i -= 1
            j += 1
    if not has_high:
        i = x
        j = y
        while i < magnitudes.shape[0] - 1 and j > 0 and labeled[i][j] == labeled[i + 1][j - 1]:
            if magnitudes[i + 1][j - 1] > t_high:
                has_high = True
                break
            i += 1
            j -= 1
    if not has_high:
        i = x
        j = y
        while i < magnitudes.shape[0] - 1 and j < magnitudes.shape[1] - 1 and labeled[i][j] == labeled[i + 1][j + 1]:
            if magnitudes[i + 1][j + 1] > t_high:
                has_high = True
                break
            i += 1
            j += 1

    k = 1 if has_high else 0

    i = x
    j = y
    while j > 0 and labeled[i][j] == labeled[i][j - 1]:
        magnitudes[i][j - 1] = k
        labeled[i][j - 1] *= k
        j -= 1
    i = x
    j = y
    while j < magnitudes.shape[1] - 1 and labeled[i][j] == labeled[i][j + 1]:
        magnitudes[i][j + 1] = k
        labeled[i][j + 1] *= k
        j += 1
    i = x
    j = y
    while i > 0 and labeled[i][j] == labeled[i - 1][j]:
        magnitudes[i - 1][j] = k
        labeled[i - 1][j] *= k
        i -= 1
    i = x
    j = y
    while i < magnitudes.shape[0] - 1 and labeled[i][j] == labeled[i + 1][j]:
        magnitudes[i + 1][j] = k
        labeled[i + 1][j] *= k
        i += 1
    i = x
    j = y
    while i > 0 and j > 0 and labeled[i][j] == labeled[i - 1][j - 1]:
        magnitudes[i - 1][j - 1] = k
        labeled[i - 1][j - 1] *= k
        i -= 1
        j -= 1
    i = x
    j = y
    while i > 0 and j < magnitudes.shape[1] - 1 and labeled[i][j] == labeled[i - 1][j + 1]:
        magnitudes[i - 1][j + 1] = k
        labeled[i - 1][j + 1] *= k
        i -= 1
        j += 1
    i = x
    j = y
    while i < magnitudes.shape[0] - 1 and j > 0 and labeled[i][j] == labeled[i + 1][j - 1]:
        magnitudes[i + 1][j - 1] = k
        labeled[i + 1][j - 1] *= k
        i += 1
        j -= 1
    i = x
    j = y
    while i < magnitudes.shape[0] - 1 and j < magnitudes.shape[1] - 1 and labeled[i][j] == labeled[i + 1][j + 1]:
        magnitudes[i + 1][j + 1] = k
        labeled[i + 1][j + 1] *= k
        i += 1
        j += 1


Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# applies the canny edge detection algorithm to an image
def myCanny(I, sigma=1, t_low=.5, t_high=1):
    # 4.1 Smooth with gaussian kernel
    I_filtered = myfilter(I, gausskernel(sigma))
    # Include this image in your report
    plt.imshow(I_filtered, cmap='gray')
    plt.axis('off')
    plt.title("Filtered Image Using a Gaussian Kernel")
    plt.show()

    # 4.2 Find img gradients
    # Convolve I with the Sobel filters
    Gx = myfilter(I_filtered, Sx)
    Gy = myfilter(I_filtered, Sy)
    # Compute image gradient direction (in degrees) and amplitude
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    # compute theta_prime by rounding theta to one of four values: 0, 45, 90, or 135
    theta_prime = convert_theta_prime(theta)
    # include the derivatives, gradient magnitude, and angles as images in your report
    plt.imshow(Gx, cmap='gray')
    plt.axis('off')
    plt.title("Derivative in x")
    plt.show()
    plt.imshow(Gy, cmap='gray')
    plt.axis('off')
    plt.title("Derivative in y")
    plt.show()
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')
    plt.title("Magnitude")
    plt.show()
    plt.imshow(theta, cmap='gray')
    plt.axis('off')
    plt.title("Angles")
    plt.show()
    plt.imshow(theta_prime, cmap='gray')
    plt.axis('off')
    plt.title("Rounded Angles")
    plt.show()

    # 4.3 Edge thinning / Non-maximum suppression
    # Thin edges
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if magnitude[i][j] == 0:
                continue
            if check_thin(i, j, theta_prime, magnitude):
                magnitude[i][j] = 0

    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')
    plt.title("Thinned Image")
    plt.show()
    # 4.4 Hystersis thresholding
    # If a pixel has a magnitude between t_low and t_high 1. identify all the pixels connected to it via a path through pixels with magnitude greater than t_low (using scipy.ndimage.measurements.label) 2. if any of the pixels in the connected component have magnitude greater than t_high, keep the edge, otherwise discard the edge
    myedges = np.zeros(magnitude.shape)
    labeled, ncomponents = label(magnitude > t_low)
    for i in range(labeled.shape[0]):
        for j in range(labeled.shape[1]):
            if magnitude[i][j] > t_high:
                myedges[i][j] = 1
            elif magnitude[i][j] < t_low:
                labeled[i][j] = 0
            elif labeled[i][j] != 0:
                check_component(i, j, labeled, myedges, t_high)
    return myedges


# 4.5 Testing
params = [(I, 1, .1, .3), (I, 1, .4, .7), (I, 1, .7, 1), (I, 2, .1, .3), (I, 2, .4, .7), (I, 2, .7, 1), (I, 3, .1, .3),
          (I, 3, .4, .7), (I, 3, .7, 1)]

for param in params:
    edges = myCanny(*param)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.title("sigma = " + str(param[1]) + ", t_low = " + str(param[2]) + ", t_high = " + str(param[3]))
    plt.show()


# Extra Credit: Hybrid Images
I1 = np.array(open('Img1.jpg').convert('L')) / 255
I2 = np.array(open('Img2.jpg').convert('L')) / 255


def low_pass(I, sigma):
    return myfilter(I, gausskernel(sigma))


def high_pass(I, sigma):
    return I - low_pass(I, sigma)


# combine the low frequencies of one image with the high frequencies of another to form a hybrid image
def hybridImage(I_1, I_2):
    I1_low = low_pass(I_1, 5)
    I2_high = high_pass(I_2, 20)
    I1_hybrid = I1_low + I2_high

    return I1_hybrid


I3 = hybridImage(I1, I2)

plt.imshow(I3, cmap='gray')
plt.axis('off')
plt.title("Hybrid Image")
plt.show()
