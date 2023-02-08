import cv2
import os

def take_photo(face_cascade):
    # Open the webcam
    cap = cv2.VideoCapture(1)

    while True:
        # Capture a single frame
        ret, frame = cap.read()

        # If the webcam was unable to capture a frame, continue to the next iteration
        if not ret:
            continue

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Show the frame
        cv2.imshow("Webcam", frame)

        # If exactly one face is detected, take a photo and break the loop
        if len(faces) == 1:
            photo = frame.copy()
            break

        # If more than one face is detected, print an error message
        elif len(faces) > 1:
            print("ERROR: More than one face detected. Please ensure that only one face is visible in the webcam.")
            exit()

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

    # Return the captured photo
    return photo

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create the dataset directory if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Take photos using the webcam until the user stops
while True:
    # Take a photo using the webcam
    photo = take_photo(face_cascade)

    # If exactly one face was detected, get the name and ID from the user and save the photo
    if photo is not None:
        name = input("Enter your name: ")
        id = input("Enter your ID: ")
        file_name = f"dataset/{name}_{id}.jpg"
        cv2.imwrite(file_name, photo)

    # Ask the user if they want to take more photos
    more = input("Take more photos? (y/n): ")
    if more.lower() != "y":
        break

print("The photos have been saved to the dataset directory.")
