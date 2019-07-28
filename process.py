import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import os
from tqdm import tqdm as tqdm

def process(video_file_name, overlay_image_name, step):

    def extract(filename, resize_to=None, step=None):
        
        class11_frames2 = []

        print('extracting video frames from',filename)

        vidcap = cv2.VideoCapture(filename)

        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print( 'number of frames',length )

        success,image = vidcap.read()
        print('video open success',success)
        print(image.shape)
        
        count = 0
        for i in tqdm(range(length)):

            if not success:
                break
                
        # skip frames to follow step
            if not step is None:
                if i % step != 0:
                    success,image = vidcap.read()
                    continue
                
            image1 = Image.fromarray(image)

            if not resize_to is None:
                image2 = image1.resize(resize_to)
            else:
                image2 = image1

            image = np.array(image2)
            class11_frames2.append(image)

            success,image = vidcap.read()
            count += 1

        class11_frames2 = np.array(class11_frames2)
        print('loaded',len(class11_frames2),'frames')

        return class11_frames2        

    def order_points(pts):
        pts = np.array(pts)
        
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = np.int32)

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def process_one_frame(frame, overlay_image, show_images = False, debug_text = False, smooth=False,
                        movie_frame_idx=None):

        imgColor = frame

        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(imgColor,cv2.COLOR_BGR2HSV)

        # Range for lower green
        lower_red = np.array([50,120,70])
        upper_red = np.array([70,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        masked = cv2.bitwise_and(hsv, hsv, mask=mask1)

        imgColorGreen = cv2.cvtColor(masked,cv2.COLOR_HSV2BGR)
        
        gray = cv2.cvtColor(imgColorGreen, cv2.COLOR_BGR2GRAY)

        ret,thresh = cv2.threshold(gray,127,255,0)
        im2,contours,hierarchy = cv2.findContours(thresh, 1, 4)
        
        areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)

            M = cv2.moments(cnt)    

        if len(areas) == 0:
            # print('empty areas')
            return None
        
        maxCnt = np.argmax(areas)
        M = cv2.moments(contours[maxCnt])

        img2 = np.zeros(imgColorGreen.shape, dtype = "uint8")
        img_mod = cv2.polylines(img2, contours[maxCnt], True, (0,255,0), thickness=3)

        cnt = contours[maxCnt]

        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        approx = [approx]
            
        if len(approx[0]) < 4:
            # print('too few corners')
            return None
            
        o = order_points(np.squeeze(np.array(approx)))
        approx = []
        approx.append([o[0]])
        approx.append([o[1]])
        approx.append([o[2]])
        approx.append([o[3]])
        approx = [approx]

        approx = np.array(approx)
        
        img2 = np.zeros([imgColorGreen.shape[0],imgColorGreen.shape[1],4], dtype = "uint8")
        img_mod = cv2.polylines(img2, approx, True, (255,0,0,255), thickness=1, lineType=8)

        img = imgColor

        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left

        # calculate transformation matrix
        pts = [[0,0],[img.shape[1],0],[img.shape[1],img.shape[0]],[0,img.shape[0]]]
        pts2 = [approx[0][0],approx[0][1],approx[0][2],approx[0][3]]
        
        if smooth and len(corners)>=1:
            
            ptsOut = []
            cornersA = np.array(corners)
            m = cornersA[-1]

            for pt_idx in range(len(pts2)):
                pt = pts2[pt_idx][0].copy()
                diff = np.round(np.abs(pt-m))

                ptOut = pt
                for dim in range(2):
                    if diff[pt_idx][dim] <= 4:
                        ptOut[dim] = m[pt_idx][dim]

                ptsOut.append([ptOut])

            # print(movie_frame_idx,np.array(ptsOut)-np.array(pts2))

            pts2 = ptsOut
            
        corners.append(np.squeeze(pts2))
        
        # compensate antialiasing green pixels
        dist = 5
        pts2 = np.array(pts2)
        pts2[0][0][0] -= dist
        pts2[0][0][1] -= dist

        pts2[1][0][0] += dist
        pts2[1][0][1] -= dist

        pts2[2][0][0] += dist
        pts2[2][0][1] += dist

        pts2[3][0][0] -= dist
        pts2[3][0][1] += dist

        pts = np.float32(pts)
        pts2 = np.float32(pts2)

        dst_pts = np.float32(pts2)

        img2 = np.zeros([imgColor.shape[0],imgColor.shape[1]], dtype = "uint8")
        img_mod = cv2.fillPoly(img2, np.array([dst_pts.astype(np.int32)]), (255))
        mask_wider = img_mod

        M = cv2.getPerspectiveTransform(pts, dst_pts)

        # wrap image and draw the resulting image
        image_size = (img.shape[1], img.shape[0])

        warped = cv2.warpPerspective(overlay_image, M, dsize = image_size, flags = cv2.INTER_LINEAR)

        mask1A = 255 - mask_wider

        imgColorMasked2 = np.moveaxis(np.array([imgColor[:,:,0],imgColor[:,:,1],imgColor[:,:,2],mask1A]),0,2)

        warpedA = 255-255*mask1A.copy()
        warpedTransp = np.moveaxis(np.array([warped[:,:,0],warped[:,:,1],warped[:,:,2],warpedA]),0,2)

        ironman = Image.fromarray(imgColorMasked2)
        bg = Image.fromarray(warpedTransp)
        text_img = Image.new('RGBA', (warpedTransp.shape[1],warpedTransp.shape[0]), (0, 0, 0, 0))
        text_img.paste(bg, (0,0))
        text_img.paste(ironman, (0,0), mask=ironman)
        if show_images:
            text_img.show()
            
        image = np.array(text_img)
        text_img.close()
        
        image2 = np.moveaxis(np.array([image[:,:,0],image[:,:,1],image[:,:,2]]),0,2)

        return image2

    frames = extract(video_file_name, step=step)

    image_size = (frames[0].shape[1], frames[0].shape[0])

    overlay_image = cv2.imread(overlay_image_name)
    overlay_image = cv2.resize(overlay_image,image_size)

    out = None
    corners = []
    output_file_name = ''

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for movie_frame_idx in tqdm(range(len(frames))):

        movie_frame = frames[movie_frame_idx]
        image = process_one_frame(movie_frame, overlay_image, smooth=True, movie_frame_idx=movie_frame_idx)
        
        if image is None:
            continue
            
        if out is None:
            output_file_name = 'output_'+str(len(frames))+'frames.mp4'
            out = cv2.VideoWriter(output_file_name, fourcc, 20.0, (image.shape[1],image.shape[0]))

        # write the frame
        out.write(image)

    print('written',output_file_name,'video')

    # Release everything if job is finished
    out.release()

# input video
video_file_name = 'input.mp4'

# process every 100th frame in video (to speed up process)
step = 100

# name of the image to overlay
overlay_image_name = 'overlay.jpeg'

process(video_file_name, overlay_image_name, step)
