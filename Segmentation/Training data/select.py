import cv2
import sys
import tkinter as tk

def get_screen_resolution():
    # Create a hidden Tkinter root window to get screen resolution
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Destroy the root window
    return screen_width, screen_height

def resize_image_for_display(img, screen_width, screen_height):
    img_height, img_width = img.shape[:2]
    
    scale_width = screen_width / img_width
    scale_height = screen_height / img_height
    scale = min(scale_width, scale_height)
    
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    img_resized = cv2.resize(img, (new_width, new_height))
    
    return img_resized, new_width, new_height

def select_and_crop_image(image_path, idx):
    test = image_path + '_test.png'
    truth = image_path + '_truth.png'
    test_img = cv2.imread(test, cv2.IMREAD_COLOR)
    truth_img = cv2.imread(truth, cv2.IMREAD_GRAYSCALE)

    if test_img is None:
        print(f"Error: Unable to load the image from {image_path}.")
        return

    screen_width, screen_height = get_screen_resolution()

    img_resized, new_width, new_height = resize_image_for_display(test_img, screen_width, screen_height)

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", new_width, new_height)

    cv2.imshow("Select ROI", img_resized)

    rect = cv2.selectROI("Select ROI", img_resized, showCrosshair=True, fromCenter=False)

    cv2.destroyWindow("Select ROI")

    # Scale the selected coordinates back to the original image size
    x, y, w, h = map(int, rect)
    scale = new_width / test_img.shape[1]
    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

    # Crop the selected region from the original images
    test_cropped_img = test_img[y:y+h, x:x+w]
    truth_cropped_img = truth_img[y:y+h, x:x+w]
    
    # Display cropped
    cv2.imshow("Cropped Image", test_cropped_img)

    # Wait for the user to press a key and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite(image_path + '_test_' + idx + '.png', test_cropped_img)
    cv2.imwrite(image_path + '_truth_' + idx + '.png', truth_cropped_img)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if len(sys.argv) > 2:
            idx = sys.argv[2]
        else:
            idx = 'sub'
        select_and_crop_image(image_path, idx)
    else:
        print("Usage: pass a name <name> as argument, and training data will be generated for <name_test.png> and <name_truth.png>.")

