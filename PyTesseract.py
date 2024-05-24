import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

CROPPED_IMAGE_PATH = "cropped_image.png"

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\Nguyen Minh Thanh\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pytesseract'

def determine_text_color(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    hist /= hist.sum()

    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_black_pixels = np.sum(thresholded_image == 0)
    num_white_pixels = np.sum(thresholded_image == 255)

    if num_black_pixels < num_white_pixels:
        text_color = "black"
        background_color = "white"
    else:
        text_color = "white"
        background_color = "black"

    print(f"Text color: {text_color}, Background color: {background_color}")

    return text_color, background_color


def remove_border(image_path, output_path):
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)

    eroded = cv2.erode(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])

    cropped_image = image[y:y+h, x:x+w]

    cv2.imwrite(output_path, cropped_image)

    return cropped_image


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text_color, background_color = determine_text_color(image_path)

    if text_color == 'black':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binary = cv2.bitwise_not(binary)

    cv2.imshow("Image After Thresholding: ", binary)
    cv2.waitKey(0)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    return image, dilated

def find_text_clusters(dilated_image):
    cv2.imshow("Dilated Image", dilated_image)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_text_from_clusters(image, contours):
    extracted_text = ""
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Draw rectangle around contour
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop the text block for OCR
        text_block = image[y:y+h, x:x+w]

        cv2.imshow("contours", text_block)
        cv2.waitKey(0)

        width = int(image.shape[1] * 120 / 100)
        height = int(image.shape[0] * 120 / 100)
        new_dimensions = (width, height)
        resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

        kernel = np.ones((2, 2), np.uint8)
        morphed_image = cv2.morphologyEx(resized_image, cv2.MORPH_DILATE, kernel)

        text = pytesseract.image_to_string(morphed_image, config='--oem 3 --psm 6')

        boxes = pytesseract.image_to_boxes(morphed_image)


        for box in boxes.splitlines():
            box = box.split(' ')
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            # Draw the box on the image
            cv2.rectangle(image, (x, morphed_image.shape[0] - y), (w, morphed_image.shape[0] - h), (0, 255, 0), 2)

        cv2.imshow('Image with boxes', morphed_image)
        cv2.waitKey(0)

        extracted_text += text + "\n"
    return extracted_text

def extract_text_from_image(image_path):
    original_image, preprocessed_image = preprocess_image(image_path)
    
    contours = find_text_clusters(preprocessed_image)

    text = extract_text_from_clusters(original_image, contours)

    print("Extracted Text:")
    print(text)

    result_image_path = 'detected_text_clusters.jpg'
    cv2.imwrite(result_image_path, original_image)

    return text