import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_random_images(data_dir):
    """
    Plots random images from the dataset.

    Parameters:
    - data_dir (str): The path to the main dataset directory.

    Returns:
    - None
    """
    # create a 10x5 grid of subplots for displaying images
    fig, ax = plt.subplots(10, 5, figsize=(40, 15))

    # loop through each digit (0 to 9)
    for digit in range(10):
        # directory for each digit's dataset
        digit_dir = os.path.join(data_dir, str(digit))

        # make a filelist of images with the '.jpg' extension
        filelist = [image for image in os.listdir(digit_dir) if image.endswith('.jpg')]

        # pick 5 random images from the current digit's dataset
        random_images = random.sample(filelist, 5)

        # display the randomly chosen images for the current digit
        for i, image_name in enumerate(random_images):
            # construct the full path to the image
            image_path = os.path.join(digit_dir, image_name)

            # read the image using matplotlib.image
            img = mpimg.imread(image_path)

            # display the image in the current subplot
            ax[digit, i].imshow(img)
            ax[digit, i].set_title(image_name)
            ax[digit, i].axis('off')

    # adjust layout for better spacing
    plt.tight_layout()

    # show the plot with all 5 images for each digit
    plt.show()
