# %%
Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
mag, ang = cv2.cartToPolar(Ix, Iy)
imshow(mag, ang, figsize=(12, 16))