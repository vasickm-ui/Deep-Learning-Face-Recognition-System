import cv2

image = cv2.imread("data/samples/face_1.jpg")

if image is None:
    print("Image not found")
    exit()

print(cv2.__file__)

# Haar Cascade is faster on gray scale images
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#this line just loads the pre-trained model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# this returns the list of tuples representing face images coordinates
# scaleFactor shrinks the image window by 10 percent each iteration 
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Number of faces detected: ", len(faces))

# draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

#saving image to results folder
cv2.imwrite("results/result.jpg", image)
print("Image saved!")

cv2.imshow("Face Detection (Color)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()