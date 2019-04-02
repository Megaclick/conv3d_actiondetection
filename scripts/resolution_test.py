import numpy as np
import cv2

cap = cv2.VideoCapture('../1.mp4')
cap.set(cv2.CAP_PROP_FPS,10)
x = cap.get(cv2.CAP_PROP_FPS)
cap.set(3,640)
cap.set(4,480)
counter = 0 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    if counter == 10:
    	# Display the resulting frame
  		cv2.imshow('frame',frame)
  		counter = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    counter+=1


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()