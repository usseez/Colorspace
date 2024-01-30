###########################################################################################################################
#################################### reduce data from BGR image to YUV422(YUYV format) ####################################
###########################################################################################################################

# Default imports
import cv2
import numpy as np
import sys

# Prepare BGR input (OpenCV uses BGR color ordering and not RGB):
img = cv2.imread('cat.bmp')

# 불러왔는지 확인
if img is None:
    print('Image load failed!')
    sys.exit()



# Split channles, and convert to float for calculation
b, g, r = cv2.split(img.astype(float))

rows, cols = r.shape

# Use BT.709 standard "full range" conversion formula
y = 0.2126*r + 0.7152*g + 0.0722*b
u = 0.5389*(b-y) + 128
v = 0.6350*(r-y) + 128

# Downsample u horizontally
print(u.shape)
u = cv2.resize(u, (cols//2, rows), interpolation=cv2.INTER_LINEAR)
print(u.shape)
# Downsample v horizontally
v = cv2.resize(v, (cols//2, rows), interpolation=cv2.INTER_LINEAR)

# Convert y to uint8 with rounding:
y = np.round(y).astype(np.uint8)

# Convert u and v to uint8 with clipping and rounding:
u = np.round(np.clip(u, 0, 255)).astype(np.uint8)
v = np.round(np.clip(v, 0, 255)).astype(np.uint8)

## create uv data
# y크기의 uv matrix 생성
uv = np.zeros_like(y)
# YUVformat : YUV422 packed format
uv[:, 0::2] = u
uv[:, 1::2] = v
print("uv.shape : ", uv.shape)
# Merge y and uv channels
yuyv = cv2.merge((y, uv))
# yuyv = np.vstack((y, uv))


# Convert yuv422 to BGR for display and saving
bgr_output = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

# Save BGR image to PNG
cv2.imwrite('yuyv_output.png', bgr_output)
cv2.imshow('yuyv_output', bgr_output)
cv2.waitKey(0)

##yuyv와 img에서  한 pixel의 bit 수 더한 값 출력
print(img.dtype)
print("img size : ", img.size)

print(yuyv.dtype)
print("yuyv size : ", yuyv.size)

img_bit = img.itemsize*8
yuyv_bit = yuyv.itemsize*8

print("data size_img (size * bit) : ", img_bit * img.size)
print("data size_yuyv (size * bit : ", yuyv_bit * yuyv.size)

# 파일 경로 및 이름 정의
yuv_file_path = '/colorextraction/workspace/Data_Reduction/result/yuyv_reduction.yuv'

# .yuv 파일로 저장
with open(yuv_file_path, 'wb') as yuv_file:
    yuv_file.write(yuyv.tobytes())
# print("u", u.shape)
# print("v", v.shape)
# print("uv", uv.shape)
# print("yuyv", yuyv.shape)

# file_size_original = os.path.getsize(img)
# file_size_yuv422 = os.path.getsize(yuyv)
# print("original file size : ", file_size_original)
# print("yuv422 file size : ", file_size_yuv422)




















# # Save Y, U, V channels of YUV422 image as separate PNG images
# cv2.imwrite('output_y_channel.png', yuv422[:, :, 0])
# cv2.imwrite('output_u_channel.png', yuv422[:, :, 1])
# cv2.imwrite('output_v_channel.png', yuv422[:, :, 2])

# # Create a 3-channel YUV image
# y_channel = cv2.imread('output_y_channel.png', cv2.IMREAD_GRAYSCALE)
# u_channel = cv2.imread('output_u_channel.png', cv2.IMREAD_GRAYSCALE)
# v_channel = cv2.imread('output_v_channel.png', cv2.IMREAD_GRAYSCALE)

# # Resize U and V channels to match Y channel
# u_channel = cv2.resize(u_channel, (y_channel.shape[1], y_channel.shape[0]))
# v_channel = cv2.resize(v_channel, (y_channel.shape[1], y_channel.shape[0]))

# # Merge Y, U, V channels
# yuv_merged = cv2.merge((y_channel, u_channel, v_channel))

# # Save the merged YUV image as PNG
# cv2.imwrite('output_yuv_merged.png', yuv_merged)




# def make_lut_u():
#     return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

# def make_lut_v():
#     return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)


# img = cv2.imread('cat.png', cv2.IMREAD_COLOR)
# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUY_422)


# y, u, v= cv2.split(img_yuv)
# b, g, r = cv2.split(img)

# lut_u, lut_v = make_lut_u(), make_lut_v()


# img_merged_bgr = cv2.merge((b,g,r))
# result = np.vstack([y, u, v])


# print(result.shape)

# cv2.imshow('y', y)
# cv2.waitKey(0)
# cv2.imshow('u', u)
# cv2.waitKey(0)
# cv2.imshow('v', v)
# cv2.waitKey(0)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.imshow('img_merged_bgr ', img_merged_bgr)
# cv2.waitKey(0)


# cv2.imwrite('img_yuv.png', img_yuv)
# # cv2.imwrite('img_merged.png', img_merged)
# cv2.imwrite('img_merged_bgr.png', img_merged_bgr)



# img_file_size = os.path.getsize('cat.png')
# img_yuv_file_size = os.path.getsize('img_yuv.png')
# img_merged_yuv_merged_file_size = os.path.getsize('img_merged.png')
# img_merged_bgr_file_size = os.path.getsize('img_merged_bgr.png')

# print('file_size_original : ', img_file_size)
# print('file_size_yuv : ', img_yuv_file_size)
# print('img_merged_yuv_merged_file_size : ', img_merged_yuv_merged_file_size)
# print('file_size_img_merged_bgr : ', img_merged_bgr_file_size)

