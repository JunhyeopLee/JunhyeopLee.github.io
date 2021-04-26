---
title: "Paper Review 1: [CVPR 2020] SampleNet: Differentiable Point Cloud Sampling"
categories:
  - Paper Review
tags:
  - CVPR
---

# 들어가며
Point cloud를 활용한 여러 논문들 대부분은 (물론 Deep Learning 활용) Farthest Point Sampling (FPS) 라고 하는 sampling 기법을 사용한다. [[FPS ref논문 link](https://arxiv.org/pdf/1612.00593.pdf)]

여러 Lidar point cloud 를 활용한 논문들 (ex. lidar scene segmentation, lidar object detection 등) 또한, backbone에 PointNet MLP 구조를 따르며, 이 때, FPS를 활용하여 sampling 된 point cloud 를 활용한다.

하지만, 이 논문에서는 이 부분에 대해서 tackle을 걸며, differentiable한 sampling 방법을 제안하며, FPS를 활용했을 때보다 성능이 좋음을 보여주며 CVPR 20 oral에 선정되었다.


<!-- # Intro  -->



