import argparse
import logging
import mouse
import cv2
import time
import threading
from tf_pose.estimator import TfPoseEstimator,Human
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.tensblur.imagemake import  image_make

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def check_fist(image,fist_show):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fist = fist_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in fist:
        if w > 0 and h > 0:
            fist_show = True
            return fist_show

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')

    cam = cv2.VideoCapture("cam1.mp4")
    #cam = cv2.VideoCapture(0)
    #cam.set(3, 1920)
    #cam.set(4, 1080)
    cam.set(5,30)
    ret_val, image = cam.read()

    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    upper_cloth = cv2.imread('cloth.png', -1)
    upper_cloth2 = cv2.imread('tshirt.png',-1)
    lower_cloth = cv2.imread("pants.png", -1)
    lower_cloth2 = cv2.imread("denim.png", -1)
    lower_cloth3 = cv2.imread("pants2.png", -1)
    fist_cascade = cv2.CascadeClassifier('fist.xml')

    upper_show = False
    lower_show = False
    fist_show = False
    whole_show = False

    while True:
        ret_val, image = cam.read()
        image = cv2.flip(cam.read()[1], 1)
        temp_h, temp_w = image.shape[:2]

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        if(whole_show):
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        try:
            #if len(humans) > 0:
                body = Human.get_upper_body_box(humans[0], temp_w, temp_h)
                lefthand_part = humans[0].body_parts[7]
                center = (int(lefthand_part.x * temp_w + 0.5), int(lefthand_part.y * temp_h + 0.5))
                cv2.circle(image, center, 5, [0, 0, 255], thickness=-1)
                #mouse.move(center[0], center[1], absolute=True, duration=0)
                #mousepoint = mouse.get_position()
                #logger.debug(str(mousepoint))

                if "w" in body and (upper_show) == True:
                    if (body["w"] > 0 and body["h"] > 0):
                        shoulder = humans[0].body_parts[2]
                        hip = humans[0].body_parts[8]
                        shoulder_point = int(shoulder.y * temp_h + 0.5) -20
                        hip_point = int(hip.y * temp_h + 0.5)

                        temp3 = int(body["x"] + body["w"] * 0.5)
                        temp4 = int(body["x"] - body["w"] * 0.5)

                        adjust_height = abs(shoulder_point - hip_point)
                        roi_color = image[shoulder_point-30:hip_point+30, temp4-30:temp3+30]
                        cloth_a = cv2.resize(upper_cloth2, (body["w"]+60, adjust_height+60), interpolation=cv2.INTER_CUBIC)
                        cv2.putText(image, 'Upper_cloth', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                        image_make.transparentOverlay(roi_color, cloth_a)


                if "w" in body and (lower_show) == True:
                    if (body["w"] > 0 and body["h"] > 0):
                        r_hip = humans[0].body_parts[8]
                        l_hip = humans[0].body_parts[11]
                        ankle = humans[0].body_parts[10]
                        hip_point_y = int(r_hip.y * temp_h + 0.5) -10
                        ankle_point_y = int(ankle.y * temp_h + 0.5)
                        temp3 = int(r_hip.x * temp_w + 0.8)
                        temp4 = int(l_hip.x * temp_w + 0.8)

                        adjust_width = (abs(temp3 - temp4))
                        adjust_height = abs(hip_point_y - ankle_point_y)
                        roi_color = image[hip_point_y-20:ankle_point_y+20, temp3-20:temp4+20]
                        cloth_a = cv2.resize(lower_cloth3, (adjust_width+40, adjust_height+40), interpolation=cv2.INTER_CUBIC)
                        cv2.putText(image, 'lower_cloth', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                        image_make.transparentOverlay(roi_color, cloth_a)



        except:
            pass
        #time.sleep(0.1)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        k= cv2.waitKey(1)
        if k == 27:
            break
        elif k & 0xFF == ord('u'):
            upper_show = not(upper_show)
        elif k & 0xFF == ord('l'):
            lower_show = not(lower_show)
        elif k & 0xFF == ord('d'):
            whole_show = not(whole_show)


        #logger.debug(str(image))
        #logger.debug('finished+')
        # #logger.debug('show+')

    cv2.destroyAllWindows()
