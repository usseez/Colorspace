# Default imports
import sys
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2


#이미지 로딩
image = cv2.imread("cat.bmp")

# 불러왔는지 확인
if image is None:
    print('Image load failed!')
    sys.exit()


# BGR to RGB function
def bgr2rgb(bgr):
    rgb = bgr[:, :, ::-1]
    return rgb

#RGB888 to RGB 565 function
def rgb2rgb565(rgb):
    height, width, channel = rgb.shape
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    
    r5 = ((r >> 3) & 0x1f) << 11
    g6 = ((g >> 2) & 0x3f) << 5
    b5 = ((b >> 3) & 0x1f)
    rgb565 = r5 | g6 | b5
    
    # for y in range(height):
    #     for x in range(width):
    #         red = (r / 255 * 31)
    #         green = (g / 255 * 31)
    #         blue = (b / 255 * 31)
    #         red5_shifted = red << 11
    #         green6_shifted = green << 5
    #         rgb565 = red5_shifted | green6_shifted | blue
    
    return rgb565


# def rgb565_2rgb(rgb565):
#     r_component = ((rgb565 >> 10) & 0x1F)
#     g_component = ((rgb565 >> 5) & 0x1F)
#     b_component = ((rgb565) & 0x1F)
#     total_compoenet = b_component | (g_component << 5) | (r_component << 10)
    
#     # 각 성분을 8비트로 변환 (optional)
#     b_component = b_component.astype(np.uint8)
#     g_component = g_component.astype(np.uint8)
#     r_component = r_component.astype(np.uint8)
#     # B, G, R 성분을 NumPy 배열로 구성
#     bgr_components = [b_component, g_component, r_component]
    
#     # BGR555_cv2_image = cv2.merge([BGR555_cv2_image_2[:,:,0], BGR555_cv2_image_2[:,:,1], np.zeros_like(BGR555_cv2_image_2[:,:,0])])
#     BGR565_cv2_image = cv2.merge([BGR565_cv2_image_golden[:,:,0], BGR565_cv2_image_golden[:,:,1], np.zeros_like(BGR565_cv2_image_golden[:,:,0])])


#     # BGR 이미지로 합치기
#     BGR555_cv2_image = cv2.merge(bgr_components)
    
    
#     return bgr_components
  
BGR565_cv2_image_golden = cv2.cvtColor(image, cv2.COLOR_BGR2BGR565)

rgb_image = bgr2rgb(image)
rgb565_image = rgb2rgb565(rgb_image)

print(rgb565_image)

##파일 저장
# 파일 경로 및 이름 정의
rgb565_image_file_path = '/colorextraction/workspace/Data_Reduction/result/rgb565_image.bin'
# goldenfile.bin 파일로 저장
with open(rgb565_image_file_path, 'wb') as bgr_file:
    bgr_file.write(rgb565_image.tobytes())








# print('BGR565_cv2_image_2 = ', BGR565_cv2_image_golden.shape)
# print('BGR565_cv2_image = ', BGR565_cv2_image.shape)


# file_size_BGR565 = os.path.getsize('BGR565_cv2_image.png')

# print('BGR565_cv2_file_size : ', file_size_BGR565)

# print('data_type of BGR565_cv2_image : ', BGR565_cv2_image.dtype)




# cv2.imshow('BGR555_cv2_image', BGR555_cv2_image)
# cv2.waitKey(0)
# cv2.imshow('BGR565_cv2_image', BGR565_cv2_image)
# cv2.waitKey(0)

