import cv2
img = cv2.imread("Images\Jump 2.png")
print(img is None)  # Om True, betyder det att bilden inte hittas
