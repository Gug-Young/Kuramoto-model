# Performace TEST
Kuramoto model을 통해 Python 과 Juli의 속도 차의를 확인해 보았다.

## 1. Kuramoto model (method RK4)
time = [0,10]  
dt = 0.01  
시간 복잡도 O(N^2)
 | N의 개수 | Python(3.10) | Julia(1.7.3) |
 | -------- | ------------ | ------------ |
 | 100      | 522ms        | 363ms        |
 | 200      | 1.76s        | 1.612s       |
 | 400      | 12.5s        | 7.635s       |
 | 800      | 49.3s        | 33.402s      |
 | 1600     | 193s         | 173.685 s    |
 | 3200     | 810s         | 640.305 s    |
 행렬 관련해서는 Julia가 Python보다는 빠르다

 ## 2. Kuramoto mean field (method RK4)
 same with previous  
 시간 복잡도 O(N)  

 | N의 개수 | Python(3.10) | Julia(1.7.3) |
 | -------- | ------------ | ------------ |
 | 100      | 90.2ms       | 28.772 ms    |
 | 200      | 114 ms       | 45.351 ms    |
 | 400      | 164 ms       | 81.719 ms    |
 | 800      | 253 ms       | 168.397 ms   |
 | 1600     | 449 ms       | 417.388 ms   |
 | 3200     | 834 ms       | 998.850 ms   |
 | 6400     | 1.56 s       | 1.789 s      |
 | 12800    | 3.13 s       | 3.865 s      |

???? 후반부에서 왜 Julia가 느려지는지는 모르겠다.
 
