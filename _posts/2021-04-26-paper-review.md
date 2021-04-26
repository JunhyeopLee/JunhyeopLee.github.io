---
title: "Paper Review 2: [NeurIPS 2019] Point-Voxel CNN for Efficient 3D Deep Learning"
categories:
  - Paper Review
tags:
  - NeurIPS
---

# Abstract
- 3D point cloud 를 위해 빠르고 효과적인 model인 Point-voxel CNN (PVCNN) 제안한 논문
- 이전의 work 들은 voxel- 또는 point-based 논문들이 주를 이루었지만, 두 방법 보두 computationally inefficient함
  ### 1. Voxel-based
  - 그도 그럴것이, voxel-based는 high-resolution data가 input일 때만 효과적임. 이유는 low-resolution이라면 point들이 뭉쳐져서 semantically 다른 point들임에도 불구하고, 한 voxel에 들어가버리는 case도 있고, (information loss)
  - resolution이 증가하면 할수록, computation cost 와 memory footprints 가 cubically 증가하기에, resolution을 높이는것이 거의 불가능함 (memory issue)
  ### 2. Point-based
  - input들어와서 feature 추출하는기까지 걸리는 시간의 약 80%가 ***sparse data 구축 (실제로는 poor memory locality를 갖는 -> 단점)*** 하는데 사용됨 (저렇게 구축하고 나서 feature 추출함)
- 이를 해소하기위해, 3D point cloud 를 위해 빠르고 효과적인 model인 Point-voxel CNN (PVCNN) 제안함 **(memory & computation efficiency)**
  - point를 활용하면서 memory consumption을 줄이고, 
  - voxel에서의 convolution을 수행하면서, irregular & sparse data access를 줄이고, locality를 좋게 만듦 
- semantic and part segmentation dataset에 대해 확인해보니, 
  - voxel-based 방법들보다 **10**X GPU memory reduction을 보이면서 높은 정확도를 보임
  - point-based 방법들보다 평균 **7**X speedup을 보임


# Motivation
1. Voxel-based models: Large Memory Footprint
![fig2a](/assets/images/2021-04-26-paper-review/fig2a.png)
- 일반적으로 voxel-based representation은 regular하고, 좋은 memory locality를 갖음
- 하지만, information loss를 줄이기위해 high-resolution을 가져야 함
- 위 그림을 보면, point들이 뭉개지지 않고, 잘 구별가능할 정도가 되려면 resolution이 커져야 하며, GPU resource가 cubically 증가하게 되기에, **voxel-based solution is not scalable!**

2. Point-based models: Irregular Memory Access and Dynamic Kernel Overhead
- 일반적으로 point-based 3D modeling 방법들은 memory efficient함 (e.g., PointNet)
- 하지만 local context modeling 능력이 떨어지기에, 후에 나온 논문들은 point domain에서의 주변 정보들을 통합/활용해서 PointNet의 표현력을 향상시킴!
- 이런 노력에도 불구하고, 이는 irregular memory access pattern을 야기하며, dynamic kernal computation overhead가 붙게 됨 -> 또한 이는 효율성측면에서 bottleneck이 됨\
![fig2b](/assets/images/2021-04-26-paper-review/fig2b.png)
- 다시 recap해보면,
  - **Irregular memory access**
    - voxel-based와 달리, $x_k$의 주변 점들은 memory상에서 인접하여 놓여있지 않다. (unordered point structure)
    - 그래서 kNN과 같은 nearest neighbors 기법을 point coordinate space에서 하거나, feature space에서 함
      - coordinate상 또는 feature space 상에서 NN을 하는 것은 expensive computation을 요구함
      - 또한, 주변점들을 모을 때, large abount of random memory access가 필요함 -> not cache friendly.
    - 위 그림을 보면, PointCNN 또는 DGCNN만 보더라도, 전체 프로세스(Irregular Access -> Dynamic Kernel -> Actual Computation for feature extraction)에서 **Irregular Access** 가 차지하는 비중이 대부분을 차지함
  - **Dynamic Kernel Computation**
    - 일반적인 2D와는 달리, point cloud에서의 point들은 irregular하게 산재해 있기 때문에, $x_k$의 주변점 $x_i$들이 each center $x_k$ 마다 달라짐. 
    - 즉, kernel K(x_k, x_i)가 매 포인트 $x_k$ 마다 계속 calculate 하는 작업 필요
    - 마찬가지, 위 그림을 보면, PointCNN의 경우, **Dynamic Kernel Computation** 가 차지하는 비중이 매우 큼
    - 합쳐서 생각해보면, 실제 feature extraction을 위한 computation하는 비중이 DGCNN(45%), PointCNN(12%) 로 매우 적음
    - 즉, point-based 방법들에서 이뤄지는 연산들의 대부분이 irregularity를 다루기 위해 사용됨 -> 비효율적임!!


# Point-Voxel Convolution
![fig3](/assets/images/2021-04-26-paper-review/fig3.png)
- 기존 voxel- 과 point-based 방법들의 bottleneck들에 대해 분석을 기반으로, hardware-efficient한 primitive를 제안함 -> Point-Voxel Convolution (PVConv)
  - point-based 방법들의 장점(small memory footprint)와 voxel-based 방법들의 장점(good data locality and regularity)를 섞음
- 즉, 위 그림처럼 2개의 brach를 통해서 voxel-based feature (**coarse-grained feature**) + point-wise feaeture (**fine-grained feature**) 를 combine함
  - upper voxel-based branch
    - 주변 point들의 정보를 활용해서 voxelize / devoxelize를 진행
    - 한번 scan해서 voxelize/devoxelize하기에, memory cost 가 낮음
      - ***원래는 voxel-based는 high-resolution을 유지하기 어려움 -> gpu resource 때문에***
      - ***그 단점을 point-based branch에서 매꿔줌***
  - lower point-based branch
    - 주변 point 정보 활용하지 않고, 개개의 point들에 대해서 feature 추출함
    - 주변 정보 활용안하니, high resolution을 유지해도 됨
      - ***원래는 주변 정보들 indexing(for NN)하는 작업들이 issue였는데, 여기에서는 그 작업을 안함***
      - ***주변 정보들 활용은 voxel-based branch에서 다룸***


# PVConv