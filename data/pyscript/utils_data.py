import cv2

def crop_face_from_scene(image, bbox, scale):
    """
    Args:
        image: np.ndarray
        bbox: [left top width height]
        scale: crop scale
    """
    y1,x1,w,h = bbox
    y2 = y1 + w
    x2 = x1 + h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region


# Training, random scale from [1.2 to 1.5]
face_scale = np.random.randint(12, 15) / 10.0
image_path = os.path.join(image_path, image_name)
image_x_temp = cv2.imread(image_path)
bbox_path = os.path.join(image_path, bbox_name)
image_x = cv2.resize(crop_face_from_scene(image_x_temp, bbox_path, face_scale), (256, 256))

# Tesing, default value:
face_scale = 1.3



if __name__ == "__main__":

