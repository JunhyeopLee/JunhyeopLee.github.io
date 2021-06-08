---
title: "Paper Review 2: [NeurIPS 2019] Point-Voxel CNN for Efficient 3D Deep Learning"
categories:
  - Paper Review
tags:
  - NeurIPS
use_math: true
---

---

# 들어가며,

- 최근 scene-level semantic segmantation task 에서 sparse convolution / sparse tensor 가 대세인거 같음
- 예를 들어 semanticKITTI benchmark 만 보더라도, 상위권에 위치해있는 방법들은 대부분 sparse convolution을 선택해서 사용하고 있음
[[SemanticKITTI Benchmark](http://www.semantic-kitti.org/tasks.html#semseg)]
- 위 링크에서 1등, 2등, 4등이 sparse convolution 사용 (ex. [[Minkwoski Engine](https://github.com/NVIDIA/MinkowskiEngine)], [[TorchSparse](https://github.com/mit-han-lab/torchsparse)])
- 따라서, 이 논문을 시작으로 sparse tensor + sparse convolution 까지 다뤄볼 예정
- 참고로 이 논문은 NeurIPS 2019 Spotlight 논문임  

---
# Abstract

- 3D point cloud 를 위해 빠르고 효과적인 model인 Point-voxel CNN (PVCNN) 제안한 논문
- 이전의 work 들은 voxel- 또는 point-based 논문들이 주를 이루었지만, 두 방법 보두 computationally inefficient함  

### 1. Voxel-based

  - 그도 그럴것이, voxel-based는 high-resolution data가 input일 때만 효과적임. 이유는 low-resolution이라면 point들이 뭉쳐져서 semantically 다른 point들임에도 불구하고, 한 voxel에 들어가버리는 case도 있고, (information loss)
  - resolution이 증가하면 할수록, computation cost 와 memory footprints 가 cubically 증가하기에, resolution을 높이는것이 거의 불가능함 (memory issue)
  
<br>

### 2. Point-based

- input들어와서 feature 추출하는기까지 걸리는 시간의 약 80%가 ***sparse data 구축 (실제로는 poor memory locality를 갖는 -> 단점)*** 하는데 사용됨 (저렇게 구축하고 나서 feature 추출함)
- 이를 해소하기위해, 3D point cloud 를 위해 빠르고 효과적인 model인 Point-voxel CNN (PVCNN) 제안함 **(memory & computation efficiency)**
  - point를 활용하면서 memory consumption을 줄이고, 
  - voxel에서의 convolution을 수행하면서, irregular & sparse data access를 줄이고, locality를 좋게 만듦 
- semantic and part segmentation dataset에 대해 확인해보니, 
  - voxel-based 방법들보다 **10**X GPU memory reduction을 보이면서 높은 정확도를 보임
  - point-based 방법들보다 평균 **7**X speedup을 보임  

---

# Motivation

1. Voxel-based models: Large Memory Footprint

![fig2a](/assets/images/2021-04-26-pvcnn-neurips/fig2a.png)
- 일반적으로 voxel-based representation은 regular하고, 좋은 memory locality를 갖음
- 하지만, information loss를 줄이기위해 high-resolution을 가져야 함
- 위 그림을 보면, point들이 뭉개지지 않고, 잘 구별가능할 정도가 되려면 resolution이 커져야 하며, GPU resource가 cubically 증가하게 되기에, **voxel-based solution is not scalable!**

2. Point-based models: Irregular Memory Access and Dynamic Kernel Overhead

- 일반적으로 point-based 3D modeling 방법들은 memory efficient함 (e.g., PointNet)
- 하지만 local context modeling 능력이 떨어지기에, 후에 나온 논문들은 point domain에서의 주변 정보들을 통합/활용해서 PointNet의 표현력을 향상시킴!
- 이런 노력에도 불구하고, 이는 irregular memory access pattern을 야기하며, dynamic kernal computation overhead가 붙게 됨 -> 또한 이는 효율성측면에서 bottleneck이 됨\
![fig2b](/assets/images/2021-04-26-pvcnn-neurips/fig2b.png)  

- 다시 recap해보면,
  - **Irregular memory access**
    - voxel-based와 달리, $x_k$의 주변 점들은 memory상에서 인접하여 놓여있지 않다. (unordered point structure)
    - 그래서 kNN과 같은 nearest neighbors 기법을 point coordinate space에서 하거나, feature space에서 함
      - coordinate상 또는 feature space 상에서 NN을 하는 것은 expensive computation을 요구함
      - 또한, 주변점들을 모을 때, large abount of random memory access가 필요함 -> not cache friendly.
    - 위 그림을 보면, PointCNN 또는 DGCNN만 보더라도, 전체 프로세스(Irregular Access -> Dynamic Kernel -> Actual Computation for feature extraction)에서 **Irregular Access** 가 차지하는 비중이 대부분을 차지함  

  - **Dynamic Kernel Computation**
    - 일반적인 2D와는 달리, point cloud에서의 point들은 irregular하게 산재해 있기 때문에, $x_k$ 의 주변점 $x_i$ 들이 each center $x_k$ 마다 달라짐. 
    - 즉, kernel K($x_k$, $x_i$) 가 매 포인트 $x_k$ 마다 계속 calculate 하는 작업 필요
    - 마찬가지, 위 그림을 보면, PointCNN의 경우, **Dynamic Kernel Computation** 가 차지하는 비중이 매우 큼
    - 합쳐서 생각해보면, 실제 feature extraction을 위한 computation하는 비중이 DGCNN(45%), PointCNN(12%) 로 매우 적음
    - 즉, point-based 방법들에서 이뤄지는 연산들의 대부분이 irregularity를 다루기 위해 사용됨 -> 비효율적임!!  

---

# Point-Voxel Convolution

![fig3](/assets/images/2021-04-26-pvcnn-neurips/fig3.png)

- 기존 voxel- 과 point-based 방법들의 bottleneck들에 대해 분석을 기반으로, hardware-efficient한 primitive를 제안함 -> Point-Voxel Convolution (PVConv)
  - point-based 방법들의 장점(small memory footprint)와 voxel-based 방법들의 장점(good data locality and regularity)를 섞음  

- 즉, 위 그림처럼 2개의 brach를 통해서 voxel-based feature (**coarse-grained feature**) + point-wise feaeture (**fine-grained feature**) 를 combine함
  - upper voxel-based branch
    - 주변 point들의 정보를 활용해서 voxelize / devoxelize를 진행
    - 한번 scan해서 voxelize/devoxelize하기에, memory cost 가 낮음
      - ***<u>원래는 voxel-based는 high-resolution을 유지하기 어려움 -> gpu resource 때문에</u>***
      - ***그 단점을 point-based branch에서 매꿔줌***
  - lower point-based branch
    - 주변 point 정보 활용하지 않고, 개개의 point들에 대해서 feature 추출함
    - 주변 정보 활용안하니, high resolution을 유지해도 됨
      - ***<u>원래는 주변 정보들 indexing(for NN)하는 작업들이 issue였는데, 여기에서는 그 작업을 안함</u>***
      - ***주변 정보들 활용은 voxel-based branch에서 다룸***  

## Voxel-Based Feature Aggregation

1. Normalization
    - 모든 point들을 0부터 1사이의 normalized된 unit sphere 공간상으로 normalization 시켜줌
    - 각각의 point 들은 {$p_k$, $f_k$} 로 구성되어 있고, $p_k$는 coordinate 정보, $f_k$는 feature 정보임
      - 즉, $p_k$ 는 normalize되어서 $\hat p_k$ (=$(\hat x_k, \hat y_k, \hat z_k)$) 로 변학, $f_k$는 feature이기에 변하지 않음 (-> voxelize 안되니까!)
  
2. Voxelization
![eq2](/assets/images/2021-04-26-pvcnn-neurips/eq2.png)

    - 위 식을 통해, normalized 된 point $\hat p_k$ (=$(\hat x_k, \hat y_k, \hat z_k)$) 는 $f_k$를 평균함으로써, voxel grid ${V_{u,v,w}}$ 로 변환됨
    - $r$은 voxel resolution
    - ![binaryindicator](/assets/images/2021-04-26-pvcnn-neurips/binaryindicator.png) 는 좌표 $\hat p_k$ 가 voxel grid $(u, v, w)$에 속하는지를 판별하는 binary indicator이다.
    - **$f_{k,c}$** 는 $\hat p_k$ 의 c 번째 channel의 feature이다.
    - $N_{u,v,w}$는 normalize factor -> voxel grid안에 포함된 point들의 수

3. Feature Aggregation
    - voxelize 한 다음에는 일반적인 3D convolution 적용시키면서 feature aggregation 실시
    - 일반적인 3D model들과 마찬가지, 매 3D convolution 다음에, batch normalization + nonlinear activation function 적용

4. Devoxelization
    - 아까 위의 그림에서 voxel-based branch와 point-based branch로 나뉜다고 하였음
    - point-based branch와 합쳐주기 위해, voxel 형식을 point cloud 형식으로 바꿔줘야하기에, devoxelization 필요
    - 단순히 nearest neighbors 방법으로 interpolation하면서 voxel-to-point mapping 할 수 있지만, 이는 같은 voxel 안에 들어있는 point 들은 계속 같은 feature를 share하게 만들기 때문에 제대로 voxel-to-point mapping이 이뤄지지 못 함
    - 그래서 각각의 point들에게 mapping 될 feature들이 distinct 하도록 하기 위해, trilinear interpolation을 통해 voxel grid를 point로 변화시킴

- 여기에서 중요한 점!, voxelization & devoxelization 모두 differentiable함  

## Point-Based Feature Transformation

- voxel-based feature aggregation branch 는 주변 정보들을 coarse granularity(coarse-grained라고 생각하면 됨, [[Granuarity 다른 설명 참고](https://lastyouth.tistory.com/4)]) 한 상태에서 융합함 (voxelization & devoxelization)
  - voxelization 자체가 information loss가 들어가는 부분이고, point coordinate를 approximation해서 voxel-grid 안에 넣는 작업이기 때문에, coarse-grained 라는 표현이 맞는거 같음
  - 또한, 위 voxel-based branch는 devoxelization등을 통해 주변 점들의 정보를 활용하여 interpolation을 진행하기 때문에, 최종적으로는 coarse grained 한 정보를 추출하는 branch라 할 수 있음
- 중요한 점은, 각각의 point feature를 finer하게 만드려면, low-resolution voxel-based 방법들만으로는 어렵다(부족하다)
- 이러한 이유로, MLP를 통해 각각의 point feature들을 추출한다.
- 간단하지만, MLP는 각각의 point들에 대해 distinct하고 discriminative한 feature를 추출할 수 있기에, high-resolution point information이 매우 critical하다!
  - corase-grained voxel-based 정보를 보완하기 위해!

## Feature Fusion

- 최종적으로는 addition을 통해서 각각의 branch에서의 서로의 단점들을 보완할 수 있다.  

---
# Summary

### voxel-based branch

- 주변점들 활용
- coarse-grained feature 추출

### point-based branch

- MLP를 통해 각각의 point들에 대한 feature 추출
- high-resolution point 정보 활용 가능
- finer-grained feature 추출  

---

# Discussion

## Efficiency: Better Data Locality and Regularity. 

### [Time Complexity]

- voxelization & devoxelization 방법
  - point들을 한번만 쫙 확인하고 voxel grid에 넣기 때문에, O(n) 복잡도
    - n은 point 수
- point-bassed 방법
  - 기존 방법들은 k개의 nearest neighbor point들을 추려내고 MLP 수행하기에, O(kn) 복잡도를 가짐 
  - 하지만 PVConv는 주변점들에 대한 정보들은 voxel에서 처리하기 때문에, O(n)복잡도만 가짐
  - 다른 방법들보다 최소 k 배 빠름


## Effectiveness: Keeping Points in High Resolution.

- point-based branch는 MLP로 구성되어 있고, 가장 큰 장점은 **주변 정보들을 다룰 수 있는 능력도 가지고 있으면서 (thanks to voxel-based branch)** network 안에서 point의 수를 계속 유지할 수 있다는 점
- PointNet++ 과 비교해보자
  - 2048 point를 가지고 한다고 하면, 
    - SA 모듈에서는 75.2 ms latency & 3.6 GB GPU memory consumption 발생
    - PVConv 에서는 25.7 ms latency & 1.0 GB GPU memory consumption 발생
    - SA 모듈에서는 최종적으로 SA모듈에서의 downsampling 때문에 685 point만 남음
      -information loss 발생
    - PVConv에서는 2048 point 그대로 남음

---

# Experiments

## Obejct Part Segmentation: ShapeNetPart

- PointNet에서 MLP를 PVConv로 바꾸고, 3D-Unet구조
  
![tab1](/assets/images/2021-04-26-pvcnn-neurips/tab1.png)
![fig4](/assets/images/2021-04-26-pvcnn-neurips/fig4.png)

## Scene Segmentation: S3DIS

- Area 5에 대해서 test 진행
- PointNet++에서 PVConv를 넣어서, PVCNN++로 만들어서 진행
  - 즉, SA 모듈 들어감
  - PointNet layer에서의 MLP 를 PVConv로 바꾼것임

![tab4](/assets/images/2021-04-26-pvcnn-neurips/tab4.png)
![fig7](/assets/images/2021-04-26-pvcnn-neurips/fig7.png)

## 3D Object Detection: KITTI

- [[Qi et al. (F-PointNet)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)] 처럼 training set 에서 val set을 만들었음
  - val 에서의 instance들은 training set에 포함되어 있지 않음
  - 위의 F-PointNet 기반으로 2가지의 PVCNN 버전 구축\
    (a). Efficient Version
    - instance segmentation network 에서 오직 MLP만 PVConv로 바꾼 것

    (b). Complete Version
    - 위에서 더 나아가서, box estimation network에서의 MLP도 PVConv로 바꾼 것

![tab5](/assets/images/2021-04-26-pvcnn-neurips/tab5.png)  

---

# Conclusion

- 이 논문에서는 fast and efficient 3D deep learning을 위한 Point-Voxel CNN (PVCNN) 제안하였음
- voxel-based branch (-> dense, regular voxel representation), point-based branch (-> sparse, irregular point representation) 도입으로 memory footprint를 줄이고, irregular memory access 영향도 줄임
- 이 논문을 통해, voxel-based convolution이 비효율적이다라는 고정관념을 깨고, 
- 빠르고 효율적인 3D deep learning을 위해, point-based & voxel-based 기반 architecture를 공동설계하는 것이 조명받기를 바란다고 함