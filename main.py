from problem1b import execution1b
import cv2
if __name__ == '__main__':
    cap = cv2.VideoCapture("Tag0.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Tag0_processed.avi', fourcc, 30, (1920, 1080))
    i = 1
    while True:

        a, frame = cap.read()
        if not a:
            break

        out.write(execution1b(frame))
        print(i)
        i += 1
    # a, frame = cap.read()
    # # cv2.imshow("RES", execution1b(frame))
    # execution1b(frame)