from getImages import *
import matplotlib.pyplot as plt
import numpy as np

input_dir = '../input/'


img_list = getImages(input_dir)

def generateGaussian(size, scaleX, scaleY):
    lower_limit = int(-((size - 1) / 2))
    upper_limit = abs(lower_limit) + 1
    ind = np.arange(lower_limit, upper_limit)
    row = np.reshape(ind, (ind.shape[0], 1)) + np.zeros((1, ind.shape[0]))
    col = np.reshape(ind, (1, ind.shape[0])) + np.zeros((ind.shape[0], 1))
    G = (1 / (2 * np.pi * (scaleX * scaleY))) * np.exp(
        -(((col) ** 2 / (2 * (scaleX ** 2))) + ((row) ** 2 / (2 * (scaleY ** 2)))))
    return G


def Gblur(size, sig, image):
    kern = generateGaussian(size, sig, sig)
    blur = cv2.filter2D(image, -1, kern)
    return blur


def downscale(image, factor, boolean):
    blur = Gblur(3, 1, image)
    h, w = blur.shape[0], blur.shape[1]
    out_h = int(h / factor)
    out_w = int(w / factor)
    if boolean:
        output = np.zeros((out_h, out_w))
    else:
        output = np.zeros((out_h, out_w, blur.shape[2]))
    for i in range(out_h):
        for j in range(out_w):
            if boolean:
                output[i, j] = image[int(i * factor), int(j * factor)]
            else:
                output[i, j, :] = image[int(i * factor), int(j * factor), :]

    output = output.astype(np.uint8)
    return output


def upscale(image, factor, boolean):

    og = image.copy()
    h, w = og.shape[0], og.shape[1]
    up_h = h * factor
    up_w = w * factor
    if boolean:
        up = np.zeros((up_h, up_w))
    else:
        up = np.zeros((up_h, up_w, og.shape[2]))
    for i in range(h):
        for j in range(w):
            x = int(i * factor)
            y = int(j * factor)
            if boolean:
                up[x, y] = og[i, j]
            else:
                up[x, y, :] = og[i, j, :]

            try:
                if boolean:
                    up[x + 1, :] = up[x, :] + ((up[x + 2, :] - up[x, :]) * 0.5)
                    up[:, y + 1] = up[:, y] + ((up[:, y + 2] - up[:, y]) * 0.5)
                else:
                    up[x + 1, :, :] = up[x, :, :] + ((up[x + 2, :, :] - up[x, :, :]) * 0.5)
                    up[:, y + 1, :] = up[:, y, :] + ((up[:, y + 2, :] - up[:, y, :]) * 0.5)
            except IndexError:
                if x + 1 < up_h and y + 1 < up_w:
                    if boolean:
                        up[x + 1, :] = up[x, :]
                        up[:, y + 1] = up[:, y]
                    else:
                        up[x + 1, :, :] = up[x, :, :]
                        up[:, y + 1, :] = up[:, y, :]
                else:
                    break

    up = (up.astype(np.uint8))
    return up


def genUnitImpulse(g_list, factor, boolean):
    temp_list = g_list.copy()
    temp_list.pop(0)
    temp_list.reverse()
    l_list = []
    for i in range(len(temp_list)):
        l_list.append(upscale(temp_list[i], factor, boolean))
    l_list.reverse()
    return l_list


def gauss_pyramid(image, factor, num_vals, boolean):
    num_vals = num_vals - 1
    og = image.copy()
    gaussian_list = []
    gaussian_list.append(og)
    for i in range(num_vals):
        og = downscale(og, factor, boolean)
        gaussian_list.append(og)

    return gaussian_list


def create_pyr(list):
    py1 = list[0].copy()
    py2 = list[1].copy()
    for i in range(2, len(list)):
        extra = abs(py2.shape[1] - list[i].shape[1])
        temp_zero = np.zeros((list[i].shape[0], extra)).astype(np.uint8)
        temp_stack = np.hstack((list[i], temp_zero))
        py2 = np.vstack((py2, temp_stack))

    if py1.shape[0] != py2.shape[0]:
        excess = abs(py1.shape[0] - py2.shape[0])
        t_zero = np.zeros((excess, py2.shape[1])).astype(np.uint8)
        py2 = np.vstack((py2, t_zero))

    out = np.hstack((py1, py2))
    return out


def gen_image(list, image):
    dim = (image.shape[1], image.shape[0])
    resized_list = []
    for i in range(len(list)):
        resized_list.append(cv2.resize(list[i], dim))
    return resized_list


def conv_list(list):
    new_list = []
    for i in range(len(list)):
        new_list.append(gray(list[i]))
    return new_list


def laplacian_pyramid(g_list, l_list):
    pyr_list = []
    for i in range(len(l_list)):
        pyr_list.append(gray(abs(g_list[i] - Gblur(3, 1, l_list[i]))))
    pyr_list.insert(len(pyr_list), gray(g_list[-1]))
    return pyr_list


def hstack_list(list):
    new_img = list[0].copy()
    for i in range(1, len(list)):
        new_img = np.hstack((new_img, list[i]))
    return new_img


def gray(image):
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gr


def log_mag_fft(image, boolean=True):
    if boolean:
        gr = image
    else:
        gr = gray(image)
    out = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gr))))
    return out


def plot_fft_h(list):
    l_plot = log_mag_fft(list[0])
    for i in range(1, len(list)):
        temp = log_mag_fft(list[i])
        l_plot = np.hstack((l_plot, temp))
    return l_plot


def get_images(image, factor, num_vals, boolean):
    g_list = gauss_pyramid(image, factor, num_vals, boolean)
    l_list = genUnitImpulse(g_list, factor, boolean)
    pyr_list = laplacian_pyramid(g_list, l_list)
    laplacian = gen_image(pyr_list, image)
    g_list = conv_list(g_list)
    gaussian = gen_image(g_list, image)
    l_hst = hstack_list(laplacian)
    g_hst = hstack_list(gaussian)
    image_stack = np.vstack((g_hst, l_hst))
    l_fft = plot_fft_h(laplacian)
    g_fft = plot_fft_h(gaussian)
    fft_stack = np.vstack((g_fft, l_fft))
    pyr_gauss = create_pyr(g_list)
    pyr_lap = create_pyr(pyr_list)
    return g_list, pyr_list, image_stack, fft_stack, pyr_gauss, pyr_lap


def cv2show_list(save, string, list):
    for i in range(len(list)):
        cv2.imshow(string + "Image with {}x{} size".format(list[i].shape[1], list[i].shape[0]),
                   list[i])
        cv2.imwrite(
            save + string + " " + str(list[i].shape[1]) + "x" + str(list[i].shape[0]) + '.jpg', list[i])
        cv2.waitKey(0)


def cv2show(save, filename, string, image):
    cv2.imshow(string, image)
    cv2.imwrite(save + filename + '.jpg', image)
    cv2.waitKey(0)


def pltshow(save, filename, image):
    plt.imshow(image)
    plt.imsave(save + filename + '.jpg', image)
    plt.show()

def reconstruction(list, factor):
    temp_list = list.copy()
    temp_list.reverse()
    new_img = temp_list[0]
    for i in range(1, len(temp_list)):
        new_img = upscale(new_img, factor, True)
        new_img = Gblur(3, 1, new_img)
        new_img = new_img + temp_list[i]
    return new_img

def calculate_error(og_image, recon_image):
    error = np.sqrt(np.sum(np.power((og_image - recon_image), 2)))
    return error

for i in range(len(img_list)):
    gaussian_list, laplacian_list, gauss_stack, fft_stack, gauss_pyr, laplacian_pyr = get_images(img_list[i], 2, 5, False)
    recon = reconstruction(laplacian_list, 2)
    save = "../output/"
    error = calculate_error(gray(img_list[i]), recon)
    print("Reconstruction Error: ", error)
    # cv2show_list(save, "Gaussian", gaussian_list)
    # cv2show_list(save, "Laplacian", laplacian_list)
    cv2show(save, "Images_stack" + "_" + str(i+1), "Images stack", gauss_stack)
    pltshow(save, "FFT_stack" + "_" + str(i+1), fft_stack)
    cv2show(save, "Gaussian_Pyramid" + "_" + str(i+1), "Gaussian Pyramid", gauss_pyr)
    cv2show(save, "Laplacian_Pyramid" + "_" + str(i+1), "Laplacian Pyramid", laplacian_pyr)
    cv2show(save, "Reconstruction" + "_" + str(i+1), "Reconstructed_Image", recon)