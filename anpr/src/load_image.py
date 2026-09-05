import cv2
import os

def load_and_show_image(image_path):
    # Read the image from disk
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load image at: {image_path}")
        return None

    print(f"✅ Loaded image: {image_path}")
    print(f"Image shape (height, width, channels): {image.shape}")

    # Show the image in a window
    cv2.imshow("ANPR - Test Image", image)
    cv2.waitKey(0)       # wait until you press a key
    cv2.destroyAllWindows()

    return image

if __name__ == "__main__":
    # Change this to match one of your test image filenames
    test_path = os.path.join("test_images", "car.jpg")
    load_and_show_image(test_path)