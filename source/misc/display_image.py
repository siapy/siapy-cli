from matplotlib import pyplot as plt
from siapy.utils.plots import display_image_with_areas


def display_spectral_image(image_name):
    fig, ax = plt.subplots()
    ax.imshow(image_display)
    plt.show()
