#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from display import model, display_instances, class_names


# In[ ]:


capture = cv2.VideoCapture(0)

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('myvideo.avi', codec, 10.0, size)

while True:
    ret, frame = capture.read()
    
    if ret:
        results = model.detect([frame], verbose=1)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        output.write(frame)
        cv2.imshow("Real Time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()

