from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import PyTesseract
import JsonUtils

# Trained Model Path
yolo_model="/Users/minhthanh/Code/AIProject/model/yolo8s.pt"
web_ui_model="/Users/minhthanh/Code/AIProject/model/WebUIFormDetectionModel.pt"
ICON_login="./image.png"

# Image Path

web_image="https://assets.justinmind.com/wp-content/uploads/2018/10/headspace-login-form.png"


# Load the custom-trained model
def load_custom_detection_model(custom_model_path):
    model = YOLO('/Users/minhthanh/Code/AIProject/model/WebUIFormDetectionModel.pt')

    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    return model


# Detect Web UI Elements
def detect_web_ui_element(custom_model_path, image_path):
    # Display the model architecture (optional)
    jsonFilePath = JsonUtils.createNewFileWithFormat('output', "json")
    model = load_custom_detection_model(custom_model_path)
    objects_detected = model(image_path)
    classes = model.names
    json = process_detected_object(classes, objects_detected, jsonFilePath)
    extract_result_info(objects_detected)
    display_total_detected_object(objects_detected)


# Optional: Extracting specific information from the results
def extract_result_info(detection_result):
    for result in detection_result:
        print("Detected objects:")
        for obj in result.boxes:
            print(f"Class: {obj.cls}, Confidence: {obj.conf}, BBox: {obj.xyxy}")


# Display total detected object
def display_total_detected_object(detection_result):
    total_object = int(len(detection_result))
    if total_object > 0:
        print("Total Object Detected: " + str(total_object))
    else:
        print("No object detected with the image!")


# Process detected object and display results
def process_detected_object(classes, results, outputFilePath):
    print("Entering Detected Object Proccessing...")

    for result in results:
        img = result.orig_img   # Get the original image

        img_copy = img.copy()
        
        for box in result.boxes:
            # Convert tensors to appropriate types
            cls = int(box.cls)  # Convert class tensor to int
            conf = float(box.conf)  # Convert confidence tensor to float
            xyxy = box.xyxy.cpu().numpy()  # Convert bounding box tensor to numpy array

            # Draw bounding box
            x1, y1, x2, y2 = map(int, xyxy[0])  # Assuming xyxy is a list of arrays, take the first array
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Split Detected Object
            detected_object = img_copy[y1:y2, x1:x2]

            # Display label
            label = f'Class: {cls}, Confidence: {conf:.2f}, Name: {classes[cls]}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(str(classes[cls]), detected_object)
            cv2.waitKey(0)

            #Save detected object
            cv2.imwrite("Result/result.png", detected_object)

            #Call Pytesseract for text recognition with specific object detected
            recogized_text = PyTesseract.extract_text_from_image("Result/result.png")
 
            # Write Detection Info into JSON file
            JsonUtils.populateDetectionInfoToJsonFile(outputFilePath, classes[cls], xyxy, conf, recogized_text)
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Show image result
        cv2.imshow("Result Detection", img_rgb)
        cv2.waitKey(0)

        # Save the image if needed
        output_path = './Result/detected_image.jpg'
        cv2.imwrite(output_path, img)

        print("Exitting Detected Object Proccessing.")

# Define main class
def main():
    detect_web_ui_element(web_ui_model, ICON_login)


if __name__ == "__main__":
    main()