# Non-AI-OCR


전반적인 이론은 ppt에서 확인할 수 있다.

알고리즘 중에 아래 알고리즘을 적용하였는데
numpy 라이브러리에서 구현이 되어있다.
![window_sliding](https://user-images.githubusercontent.com/48349693/156313088-b206e22f-5a5a-4381-82f0-b1c4f1a8a78e.gif)

```python
aa = np.arange(20).reshape(1,4,5)
np.lib.stride_tricks.sliding_window_view(aa, (2,2), axis=(1,2)).shape
```


https://user-images.githubusercontent.com/48349693/156312465-de435573-1609-4c56-a731-f0e804e9a9e1.mp4

