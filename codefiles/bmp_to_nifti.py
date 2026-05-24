import os
import numpy as np
from PIL import Image
import nibabel as nib

def convert_bmp_to_nifti(bmp_dir, output_file):
    bmp_files = sorted([f for f in os.listdir(bmp_dir) if f.endswith('.bmp')])
    if not bmp_files:
        print(f"No BMP files found in {bmp_dir}")
        return

    # Read all images to get max dimensions
    images = [Image.open(os.path.join(bmp_dir, f)).convert('L') for f in bmp_files]
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    num_slices = len(images)

    # Initialize a 3D numpy array to hold the stacked images
    volume = np.zeros((max_height, max_width, num_slices), dtype=np.uint8)

    # Load BMP images into the 3D numpy array
    for i, img in enumerate(images):
        # Pad smaller images if necessary
        if img.width < max_width or img.height < max_height:
            new_img = Image.new('L', (max_width, max_height), 0)
            new_img.paste(img, (0, 0))
            img = new_img
        volume[:, :, i] = np.array(img)

    # Convert the numpy array to NIfTI format
    nifti_img = nib.Nifti1Image(volume, np.eye(4))

    # Save the NIfTI file
    nib.save(nifti_img, output_file)

    print(f'Converted {num_slices} BMP images to NIfTI format and saved as {output_file}')

# Main execution
main_dir = r'D:\OneDrive_2025-03-05\Digit Automation\Digit_Training_dataset\masks'
output_dir = r'D:\OneDrive_2025-03-05\Digit Automation\Digit_Training_dataset\masks\nifti'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for sample_name in os.listdir(main_dir):
    sample_dir = os.path.join(main_dir, sample_name)
    if os.path.isdir(sample_dir):
        output_file = os.path.join(output_dir, f"{sample_name}.nii")
       
        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {sample_name} as it has already been processed.")
            continue
       
        convert_bmp_to_nifti(sample_dir, output_file)

print("Processing completed.")





# BMP to NIFTI for masks (binary 0 and 1)
import os
import numpy as np
from PIL import Image
import nibabel as nib

def convert_bmp_to_nifti(bmp_dir, output_file):
    bmp_files = sorted([f for f in os.listdir(bmp_dir) if f.endswith('.bmp')])
    if not bmp_files:
        print(f"No BMP files found in {bmp_dir}")
        return

    # Read all images to get max dimensions
    images = [Image.open(os.path.join(bmp_dir, f)).convert('L') for f in bmp_files]
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    num_slices = len(images)

    # Initialize 3D numpy array for binary mask
    volume = np.zeros((max_height, max_width, num_slices), dtype=np.uint8)

    # Load and process BMP images
    for i, img in enumerate(images):
        # Pad smaller images if necessary
        if img.width < max_width or img.height < max_height:
            new_img = Image.new('L', (max_width, max_height), 0)
            new_img.paste(img, (0, 0))
            img = new_img
           
        # Convert 255 values to 1 while maintaining 0 background
        img_array = np.array(img)
        img_array[img_array == 255] = 1  # Key modification for binary masks
        volume[:, :, i] = img_array

    # Create and save NIfTI image
    nifti_img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti_img, output_file)
    print(f'Converted {num_slices} BMP masks to NIfTI with 0/1 values: {output_file}')

# Main execution (unchanged)
main_dir = r'E:\OneDrive_2025-03-05\Digit Automation\Digit_Training_dataset\masks'
output_dir = r'E:\OneDrive_2025-03-05\Digit Automation\Digit_Training_dataset\masks\nifti'

os.makedirs(output_dir, exist_ok=True)

for sample_name in os.listdir(main_dir):
    sample_dir = os.path.join(main_dir, sample_name)
    if os.path.isdir(sample_dir):
        output_file = os.path.join(output_dir, f"{sample_name}.nii")
       
        if os.path.exists(output_file):
            print(f"Skipping {sample_name} as it has already been processed.")
            continue
       
        convert_bmp_to_nifti(sample_dir, output_file)

print("Mask conversion completed.")
