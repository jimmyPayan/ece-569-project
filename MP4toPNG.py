# Ryan Raad 2025
# The script we used to convert the video to images for the dataset.
# Extracts frames from a video file and saves them as PNG images.
import cv2
import os
import sys

def extract_frames(video_path, output_dir='./img'):
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        sys.exit(1)

    frame_count = 0
    image_paths = []

    # Loop through all frames in the video.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Construct the filename and full path.
        filename = f"img{frame_count}.png"
        filepath = os.path.join(output_dir, filename)
        # Save the frame as a PNG file.
        cv2.imwrite(filepath, frame)
        # Append the file path in the required format.
        image_paths.append(f"./{output_dir}/{filename}")
        frame_count += 1

    cap.release()

    # Write the list of image paths to image.txt.
    with open("image.txt", "w") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"Extracted {frame_count} frames to the '{output_dir}' directory.")
    print("Image paths have been written to 'image.txt'.")

if __name__ == "__main__":
    # Check for proper usage.
    if len(sys.argv) < 2:
        print("Usage: python MP4toPNG.py <input_video.mp4>")
        sys.exit(1)

    input_video = sys.argv[1]
    extract_frames(input_video)
