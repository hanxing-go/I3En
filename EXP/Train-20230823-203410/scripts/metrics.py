import os
import piq

# Set the path to the images folder
images_folder = 'path/to/images'

# Loop through all the images in the folder
for filename in os.listdir(images_folder):
    # Load the image
    image = piq.imread(os.path.join(images_folder, filename))

    # Convert the image to tensor
    image_tensor = piq.img_tensor(image)

    # Calculate NIQE
    niqe = piq.niqe(image_tensor)

    # Calculate BRL
    brl = piq.brl(image_tensor)

    # Calculate PI
    pi = piq.pi(image_tensor)

    # Print the results
    print(f'Image: {filename}')
    print(f'NIQE: {niqe:.4f}')
    print(f'BRL: {brl:.4f}')
    print(f'PI: {pi:.4f}')