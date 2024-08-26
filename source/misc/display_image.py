from matplotlib import pyplot as plt


def display_spectral_image(image_name):
    fig, ax = plt.subplots()
    ax.imshow(image_display)
    plt.show()
