---
title: "Paper Review 3: [CVPR 2019] 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks
"
categories:
  - Paper Review
tags:
  - CVPR
---

---

# 들어가며,
- point cloud는 대개, 수천,수만개의 point들의 집합이므로, 빠르고 효율적인 network 구조가 필요
- 앞선 논문에서는 point-based & voxel-based를 같이 쓰면 효율적으로 processing 할 수 있다고 하는데,
- 이 논문에서는 Sparse Convolution을 도입하면서 효율적인 네트워크 구조를 제안함
- 참고로 Minkowski 란 말은 phisics에서 사용되는 말이며, Minkowski space(space-time continum)라는 말에서 따온 것임

---

<br>
<br>
<br>

---
# Abstract