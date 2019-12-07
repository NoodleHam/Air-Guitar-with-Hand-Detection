import cv2
import numpy as np

import simpleaudio as sa

hand_hist = None
traverse_point = []
traverse_point_2 = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    if len(contour_list) == 0:
        return None
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

    return contour_list[max_i]


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (34, 34))
    disc = np.float32(disc) # this disc is for ignoring noise
    disc /= np.count_nonzero(disc) / 2 #normalize filter by size
    cv2.filter2D(dst, -1, disc, dst)

    # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32, 32))
    # disc = np.float32(disc) # this disc is for dilation
    # disc /= np.count_nonzero(disc) * 4  # normalize filter by size
    # cv2.filter2D(dst, -1, disc, dst)
    # cv2.imshow('after filter', dst)
    # thresh = dst

    ret, thresh = cv2.threshold(dst, 4, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None)
    thresh = cv2.merge((thresh, thresh, thresh))
    # cv2.imshow('after filter', thresh)

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

def farthest_points(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)
        dist[dist_max_i] = 0
        dist_second_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_point_2 = None
            if dist_second_max_i < len(s):
                farthest_defect_2 = s[dist_second_max_i]
                farthest_point_2 = tuple(contour[farthest_defect_2][0])

            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return [farthest_point, farthest_point_2]
        else:
            return [None, None]
    else:
        return [None, None]


def draw_circles(frame, traverse_point, color=None):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            if color is not None:
                cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), color, -1)
            else:
                cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)



def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    # cv2.imshow('after filter', hist_mask_image
    contour_list = contours(hist_mask_image)
    if (len(contour_list) == 0):
        return
    max_cont = max_contour(contour_list)
    cv2.drawContours(frame, [max_cont], -1, 0xFFFFFF, thickness=4)


    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_points = farthest_points(defects, max_cont, cnt_centroid)
        # print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_points[0]))
        cv2.circle(frame, far_points[0], 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_points[0])
            traverse_point_2.append(far_points[1])
        else:
            traverse_point.pop(0)
            traverse_point_2.pop(0)
            traverse_point.append(far_points[0])
            traverse_point_2.append(far_points[1])

        draw_circles(frame, traverse_point)
        if far_points[0] is not None:
            freq = int(far_points[0][0]) * 2 # treat x location as frequency
            volume = float(far_points[0][1] / 800) # treat y location as volume
            print("Playing frequency {} Hz at volume {}".format(freq, round(volume*100) / 100.))
            play_sound(freq, volume)
        # draw_circles(frame, traverse_point_2, color=[0,0,255])


def play_sound(frequency, volume):
    frequency = frequency  # Our played note will be 440 Hz
    fs = 44100  # 44100 samples per second
    seconds = 0.05  # Note duration of 3 seconds

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * fs, False)

    # Generate a 440 Hz sine wave
    note = np.sin(frequency * t * 2 * np.pi)

    # Ensure that highest value is in 16-bit range
    audio = note * (2 ** 15 - 1) / np.max(np.abs(note)) * volume
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, fs)

    # Wait for playback to finish before exiting
    # play_obj.wait_done()

def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)

        else:
            frame = draw_rect(frame)

        cv2.flip(frame, 1, frame)
        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
