import cv2
import threading
class image_make():

    def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape  # Size of foreground
        rows, cols, _ = src.shape  # Size of background Image
        y, x = pos[0], pos[1]  # Position of foreground/overlay image

        # loop over all pixels and apply the blending equation
        for i in range(h):
            for j in range(w):
                if x + i >= rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
                src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
        return src


    # def uppercloth_make(self,humans,body,temp_h,image):
    #     if (body["w"] > 0 and body["h"] > 0):
    #         shoulder = humans[0].body_parts[2]
    #         hip = humans[0].body_parts[8]
    #         shoulder_point = int(shoulder.y * temp_h + 0.5)
    #         hip_point = int(hip.y * temp_h + 0.5)
    #         temp1 = int(body["y"] + body["h"] * 0.5)
    #         temp2 = int(body["y"] - body["h"] * 0.5)
    #         temp3 = int(body["x"] + body["w"] * 0.5)
    #         temp4 = int(body["x"] - body["w"] * 0.5)
    #         # temp5 = int(face["y"] + face["h"] * 0.5)
    #         # temp6 = int(face["y"] - face["h"] * 0.5)
    #         adjust_height = abs(shoulder_point - hip_point)
    #         roi_color = image[shoulder_point:hip_point, temp4:temp3]
    #         cloth_a = cv2.resize(upper_cloth, (body["w"], adjust_height), interpolation=cv2.INTER_CUBIC)
    #         self.transparentOverlay(roi_color, cloth_a)
