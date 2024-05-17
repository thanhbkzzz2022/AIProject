from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import PyTesseract 

# Load the custom-trained model
# model = YOLO('Users/minhthanh/Code/AIProject/best.pt')

model = YOLO('best.pt')

# image_path="/Users/minhthanh/Code/AIProject/image.png"

# text_path="./text_to_ocr.png"

# plate_number="./image copy.png"


web_image="https://assets.materialup.com/uploads/3599159f-9e56-4a90-8f51-118a74da25e3/preview.jpg"


# Display the model architecture (optional)
results = model(web_image)

# results.print()  # Print the results
# results.save(save_dir='output')  # Save the results in the specified directory

# Optional: Extracting specific information from the results
for result in results:
    print("Detected objects:")
    for obj in result.boxes:
        print(f"Class: {obj.cls}, Confidence: {obj.conf}, BBox: {obj.xyxy}")

for result in results:
    img = result.orig_img  # Get the original image
    
    total_object = len(results)

    print("There is no object detected! - Total Object Detected: " + str(total_object))

    for box in result.boxes:
        # Convert tensors to appropriate types
        cls = int(box.cls)  # Convert class tensor to int
        conf = float(box.conf)  # Convert confidence tensor to float
        xyxy = box.xyxy.cpu().numpy()  # Convert bounding box tensor to numpy array

        print("xyxy structure:", xyxy)  # Print xyxy to understand its structure

        # Draw bounding box
        x1, y1, x2, y2 = map(int, xyxy[0])  # Assuming xyxy is a list of arrays, take the first array
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Split Detected Object
        detected_object = img[y1:y2, x1:x2]

        cv2.imwrite("result.png", detected_object)

        PyTesseract.image_to_text("result.png")

        # Display label
        label = f'Class: {cls}, Confidence: {conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Test", detected_object)
        cv2.waitKey(0)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.show()

    # Save the image if needed
    output_path = 'detected_image.jpg'
    cv2.imwrite(output_path, img)

    print("End Execution")