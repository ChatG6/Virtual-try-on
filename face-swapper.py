import cv2
import numpy as np

def swap_faces(source_image, destination_image):
    # Load the source and destination images
    source_img = cv2.imread(source_image)
    destination_img = cv2.imread(destination_image)

    # Convert the images to grayscale
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    destination_gray = cv2.cvtColor(destination_img, cv2.COLOR_BGR2GRAY)

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "face-detection-model.xml")

    # Detect faces in the source and destination images
    source_faces = face_cascade.detectMultiScale(source_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    destination_faces = face_cascade.detectMultiScale(destination_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if faces are detected in both images
    if len(source_faces) == 0 or len(destination_faces) == 0:
        print("No faces detected in one or both images.")
        return None

    # Extract the first face from the source and destination images
    source_face = source_img[source_faces[0][1]:source_faces[0][1]+source_faces[0][3], source_faces[0][0]:source_faces[0][0]+source_faces[0][2]]
    destination_face = destination_img[destination_faces[0][1]:destination_faces[0][1]+destination_faces[0][3], destination_faces[0][0]:destination_faces[0][0]+destination_faces[0][2]]

    # Resize the source face to the size of the destination face
   source_face = cv2.resize(source_face, (destination_face.shape[1], destination_face.shape[0]))

    # Create a mask for the source face
    source_mask = np.zeros_like(destination_face)
    source_mask[0:source_face.shape[0], 0:source_face.shape[1]] = 255

    # Create a mask for the destination face
    destination_mask = np.zeros_like(destination_face)
    destination_mask[0:destination_face.shape[0], 0:destination_face.shape[1]] = 255

    # Replace the destination face with the source face
    destination_face[0:source_face.shape[0], 0:source_face.shape[1]] = source_face

    # Create a new image with the swapped faces
    swapped_img = cv2.add(cv2.multiply(destination_img, destination_mask), cv2.multiply(destination_face, source_mask))

    # Save the swapped image
    cv2.imwrite("swapped_image.jpg", swapped_img)

    # Display the swapped image
    cv2.imshow("Swapped Image", swapped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the swap_faces function with the paths to the source and destination images
swap_faces("source_image.jpg", "destination_image.jpg")