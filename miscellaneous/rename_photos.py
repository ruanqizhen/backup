import os
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime


def get_exif_creation_time(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is not None:
            for tag, value in exif_data.items():
                decoded_tag = TAGS.get(tag, tag)
                if decoded_tag == 'DateTimeOriginal':
                    return value
        return None
    except Exception as e:
        print(f"Error reading EXIF data from {image_path}: {e}")
        return None


def rename_images_with_exif_creation_time(folder_path):
    # List all files in the given folder
    files = os.listdir(folder_path)

    # Process each file
    for filename in files:
        file_path = os.path.join(folder_path, filename)

        # Ensure it's a file and has a .jpg or .jpeg extension
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg')):
            try:
                # Get the creation time from EXIF data
                creation_time_str = get_exif_creation_time(file_path)

                if creation_time_str:
                    # Convert EXIF date format (YYYY:MM:DD HH:MM:SS) to desired format (YYYYMMDD_HHMMSS)
                    creation_time = datetime.strptime(creation_time_str, '%Y:%m:%d %H:%M:%S')
                    formatted_creation_time_str = creation_time.strftime('%Y%m%d_%H%M%S')

                    # Extract the file extension
                    file_extension = os.path.splitext(filename)[1]

                    # Create the new filename
                    new_filename = f"{formatted_creation_time_str}{file_extension}"
                    new_file_path = os.path.join(folder_path, new_filename)

                    # Rename the file
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {filename} to {new_filename}")
                else:
                    print(f"No EXIF creation time found for {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")


# Example usage
folder_path = 'path_to_your_folder'  # Replace with the path to your folder
rename_images_with_exif_creation_time(r"C:\projects\ptyon_test")
