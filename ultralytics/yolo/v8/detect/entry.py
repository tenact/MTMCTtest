import cv2




p1 = 50 # x value
p2 = 1230 #x value
q1 = 10 #y value
q2 = 150 #y value

# Exit Plain
v1 = 50 # x value
v2 = 1230 #x value
f1 = 710  
f2 = 610


image = cv2.imread('bild2.jpg')



overlay = image.copy()
cv2.rectangle(overlay, (p1, q1), (p1+(p2-p1), q1+(q2-q1)), (0, 0, 255, 128), -1)
cv2.rectangle(overlay, (v1, f1), (v1+(v2-v1), f1+(f2-f1)), (0, 0, 255, 128), -1)

alpha = 0.5
image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)

# Display the image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()