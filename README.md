# Non-AI-OCR


전반적인 이론은 ppt에서 확인할 수 있다.

사전에 실행하기 전에 감지할 7세그먼트의 박스 위치들과  
그에 대한 7세그먼트 이미지 10장이 필요하다. (0~9)

---

##### 방법
이미지에서 박스로 crop한 이미지가 사전에 준비한 10장 이미지에서 어떤 이미지와 가장 차이가 적은가로 숫자를 판별한다.  
crop한 이미지가 사전이미지와 몇픽셀 빗나가서 인식이 잘 안될 수 있기에  
아래 알고리즘을 사용하였다.

window sliding algorithm 은 numpy 라이브러리에서 구현이 되어있다.

![window_sliding](https://user-images.githubusercontent.com/48349693/156313088-b206e22f-5a5a-4381-82f0-b1c4f1a8a78e.gif)

```python
aa = np.arange(20).reshape(1,4,5)
np.lib.stride_tricks.sliding_window_view(aa, (2,2), axis=(1,2)).shape
>> (1,3,4,2,2)
```

---

##### 사전이미지에서 일치하는 이미지 찾기
```python
import numpy as np
import cv2

def shift_window_pred(crop_imgs, images): # (3, h+20, w+20), (10, h, w)
    _, h, w = images.shape
    crop_imgs = np.lib.stride_tricks.sliding_window_view(crop_imgs, (h,w), axis=(1,2)) # (3,21,21,h,w)
    crop_imgs = crop_imgs[:, None, ...] # (3,1,21,21,h,w)
    images = images[None, :, None, None, ...] # (1,10,1,1,h,w)
    temp = np.abs(crop_imgs - images) # (3,10,21,21,h,w)
    temp = np.sum(temp, axis=(-1,-2)) # (3,10,21,21)
    temp = np.min(temp, axis=(-1,-2)) # (3,10)
    pred_nums = np.argmin(temp, axis=-1)
    return pred_nums
```
패딩한 crop이미지들과 사전이미지를 입력하면 예측된 숫자들이 나온다.  
numpy의 broadcasting 기능을 적극 활용하였다.

---

##### 메인
```python
import cv2
import time
import numpy as np
from IPython.display import clear_output

cap = cv2.VideoCapture("./video/digital_part2.mp4")
# 동영상 저장용
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,height))

images = list(map(lambda x:cv2.imread(f'./data/{x}.jpg'), range(10),))
images = list(map(lambda x:cv2.inRange(x, (0,0,128), (255,255,255), ), images))
images = np.stack(images).astype(np.int32) # (10, H, W)
_, H, W = images.shape

# xyxy
boxes = [[553,287,593,357],
         [600,287,640,357],
         [645,290,685,360],
        ]

font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    crop_imgs = list(map(lambda x:img[x[1]:x[3],x[0]:x[2]], boxes))
    crop_imgs = list(map(lambda x:cv2.resize(x, (W,H)), crop_imgs))
    crop_imgs = list(map(lambda x:cv2.inRange(x, (0,0,128), (255,255,255), ), crop_imgs))
    crop_imgs = list(map(lambda x:cv2.copyMakeBorder(x, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0]), crop_imgs))
    crop_imgs = np.stack(crop_imgs).astype(np.int32) # (3, H+20, W+20)
    pred_nums = shift_window_pred(crop_imgs, images) # (3,)

    for i, box in enumerate(boxes):
        img = cv2.rectangle(img, box[:2], box[2:], (0,255,0), 3)
        cv2.putText(img, f"{pred_nums[i]}", box[:2], font, 1, (0,0,255), 2, cv2.LINE_AA)

    # 동영상 저장용
    # out.write(img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # time.sleep(0.2)

# 동영상 저장용
# out.release()
cap.release()
cv2.destroyAllWindows()
```
이미지의 preprocessing 코드가 담겨있다.  
위에 이미지 resize하는 부분이 있는데 numpy는 rectangle 형태의 데이터만 가능하기 때문에 적용한 것이다.  
resize보다는 패딩후 이미지를 일부 잘라서 크기를 맞추는 방법이 바람직하다.

---

아래는 output 영상이다.

https://user-images.githubusercontent.com/48349693/156312465-de435573-1609-4c56-a731-f0e804e9a9e1.mp4

