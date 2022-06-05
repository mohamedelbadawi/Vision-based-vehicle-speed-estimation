





# Importing needed libraries
import numpy as np
import cv2
import time


camera = cv2.VideoCapture("video.mp4")


h, w = None, None

#     labels = [line.strip() for line in f]

labels=['car']

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3-tiny.cfg',
                                     'yolo-coco-data/yolov3-tiny.weights')

layers_names_all = network.getLayerNames()


layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]


probability_minimum = 0.3


threshold = 0.3


colours = (0, 255,0)






while True:

    _, frame = camera.read()

    # ROI
    roi = frame[300:1500, 400:1500]


    if w is None or h is None:
        h, w = roi.shape[:2]

    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)


    network.setInput(blob)  # setting blob as input to the network
    # for debug
    # start = time.time()
    output_from_network = network.forward(layers_names_output)
    # end = time.time()




    bounding_boxes = []
    confidences = []
    class_numbers = []


    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:

                box_current = detected_objects[0:4] * np.array([w, h, w, h])


                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)


    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours


            cv2.rectangle(roi, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            center_cordinates=((x_min+x_min+box_width)//2,(y_min+y_min+box_height)//2)
            red_color=(0,0,255)
            cv2.circle(roi,center_cordinates,1,red_color,3)

            text_box_current = "car"

            cv2.putText(roi, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    cv2.imshow('ROI', roi)
    cv2.imshow('full frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
