import json

OUTPUT_BASE_DIR="./Result"

# Function to create new file with specific file type
def createNewFileWithFormat(fileName, fileType):
    filePath = f"{OUTPUT_BASE_DIR}/{fileName}.{fileType}"
    with open(filePath, "w+") as file:
        file.write("")
    return filePath


# Function to append data to a JSON file
def appendDataToJson(file_path, data):
    try:
        # Load existing JSON data from the file
        with open(file_path, 'r') as file:
            json_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is empty, initialize with an empty list
        json_data = []

    # Convert the JSON-formatted string to a Python dictionary
    # print("DATA: " + str(data))
    # new_data = json.loads(data)

    # Append the new data to the existing data
    json_data.append(data)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

def populateDetectionInfoToJsonFile(filePath, element, coordinates, confidence, text):
    jsonData =  {
                    "element": f'{element}',
                    "coordinates": f'{coordinates}',
                    "confidence": f'{confidence}',
                    "detected_text": f'{text}'
                }
    appendDataToJson(filePath, jsonData)