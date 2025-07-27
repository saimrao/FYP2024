import cv2
import matplotlib.pyplot as plt
import numpy as np


def setup_display():
    plt.rc('axes', labelsize=20)
    return plt.figure(figsize=(40, 40))

def import_and_process_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (550, 820))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def calculate_saliency(image, saliency_detector):
    success, saliency_map = saliency_detector.computeSaliency(image)
    return (saliency_map * 255).astype("uint8")

def process_images(paths):
    fig = setup_display()
    images = [import_and_process_image(path) for path in paths]
    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    saliency_maps = [calculate_saliency(image, saliency_detector) for image in images]
    means = [np.mean(smap) for smap in saliency_maps]
    display_Heat_maps(fig, saliency_maps, means)

def display_Heat_maps(fig, maps, means):
    for i, (map, mean) in enumerate(zip(maps, means), 1):
        ax = fig.add_subplot(1, len(maps), i)
        ax.set_title('Saliency Map {}'.format(i))
        ax.set_xlabel('Average: {:.2f}'.format(mean), fontsize=10)
        ax.imshow(map)
    plt.subplots_adjust(wspace=0.5)
    plt.show()

def main():
    image_path1 = input('Enter the link to the first image file: ')
    image_path2 = input('Enter the link to the second image file: ')
    process_images([image_path1, image_path2])

if __name__ == "__main__":
    main()
