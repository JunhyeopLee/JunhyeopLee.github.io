---
title: "Sparse Tensor Networks at Minkoski Engine"
categories:
  - Paper Review

---

---

# 들어가며,
- Choy 논문[[학위논문](https://node1.chrischoy.org/data/publications/thesis/ch4_sparse_tensor_network.pdf)]와 [[Minkowski Engine](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html)] document 사이트, [[4D Spatio-Temporal ConvNets: Minkowski Convolutional NEural Networks, CVPR 2019](https://arxiv.org/pdf/1904.08755.pdf)]를 참고하였음
- spatially dense 한 language 또는 image와는 다르게, 3D point cloud 또는 higher-dimensional data(e.g., data statistics)들은 공간상에서 data 분포도가 매우 sparse 함
- 효율적인 learning을 위해서는 sparse representation을 어떻게 잘 활용하느냔가 관건임
- 이를 해결하기 위해 spatially sparse data를 활용해서 spatially-sparse convolutional neural network들이 개발되고, 이러한 network들은 spatially sparse tensor를 도입하고, 이에 맞는 sparse tensor를 홀용한 activation 함수들도 제안됨
- 이러한 network들은 <u>**Sparse Tensor Network**</u>이라 불리며, network에 들어가는 input 을 포함한 모든 것들이 sparse tensor로 구성되어 있음
---


# 주요 용어 및 Sparse Conv 배경지식
- 참고[[MinkowskiEngine Doc](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html)]
## Sparse Convolution
- 초기에 neural network의 inference 속도를 올리고, memory footprint를 줄이기 위해서 compression 시키는 방법들이 많이 제안되었는데, 대표적으로 [[Sparse Convolutional Neural Networks, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf)]가 있음
  - 여기에서도 Sparse Convolution이란 terminology를 사용하는데, 이는 모델 경량화할 때, weight를 pruning하는 방법임
  - 이는 parameter-space에서의 sparsity를 고려한 것이며, 실제는 dense tensor를 활용한 연산임
  - data-space에서의 sparsity를 고려한 방법이 아님

- 여기에서의 Sparse Convolution은 **spatially-sparse data**를 기반으로 함
- 이러한 data를 다루기 위해 sparse tensor의 개념을 도입
  - 이 sparse convolution 논문에서 개념 도입: [[3D Semantic Segmentation with
Submanifold Sparse Convolutional Networks, CVPR 2018](https://arxiv.org/pdf/1711.10275.pdf)]



<br>

---
# Sparse Tensor Networks
- 일반적인 convolutional network(dense tensor) 들하고의 차이점은 sparsity management에 있음

<br>

| 구분 | Conventional network | Sparse Tensor Network |
|---|:---:|---:|
| `Tensor` | Dense tensor | Sparse tensor |
| `Random Access` | Easy(e.g., pixel grid) | Difficult(e.g., require hash-table or KD-tree) |

<br>


- 위 table 처럼, 무작위로 배치되어 있는 element에 접근하는것은
  - dense tensor의 경우에는 쉽지만,
  - sparse tensor의 경우에는 상대적으로 복잡한 data structure를 담고 있는 무언가가 필요함 (hash-table 또는 KD-tree, 또는 간단하게 KNN)
- 즉, sparse data를 다룰 때, convolution 연산을 위해 주변점 찾는 것 + 간단하게 max-pooling하는것은 더이상 trivial operation이 아니게 된다는 점


<br>
<br>

---

## Terms
### 1. Sparse Tensor
- 대부분의 element들이 0인 sparse tensor (<- sparse matrix의 확장판이라고 보면 됨)
- $x^d_i$는 data coordinate이고, 이 값들이 C 안에 들어가 있다면 $f_i$ 값을 갖고 나머지는 0

![eq1](/assets/images/2021-04-27-sparse-tensor/eq1.png)

  - COO format: Coordinate list
![eq2](/assets/images/2021-04-27-sparse-tensor/eq2.png)

<br>

### 2. Tensor Stride
image stride 개념과 같음

<br>

### 3. Kernel Map
![fig2](/assets/images/2021-04-27-sparse-tensor/fig2.png)

- sparse tensor 가 다른 sparse tensor 에 mapping 될 때(convolution 또는 pooling layer지나면서), 어떤 coordinate가 mapping되었는지를 알아야 함
- 위 그림처럼 3X3 kernel을 활용한다고 가정하면, sparse tensor(input map, $I$)에 kernel을 적용하면, $I$에서의 3X3에 위치하는 정보들이 sparse tensor (output map, $O$)의 한 위치에 mapping 되는 것이니, 총 9개의 map (A, B, C, D, E, F, G: kernel의 각 위치)이 존재하는 것임
- 위 그림을 정리해보면
  - I:0→0, B:1→0, B:0→2,, D:3→1,, H:2→3 ($I$→$O$)
- coordinate mapping, ($I → O$): $I$에서의 integer index를 활용해서, 어떤 위치에 있는 coordinate가 다음 tensor $O$의 integer index에 mapping되는지 list-up
- mapping list를 통해 $F_I$ 가 $F_O$에 mapping 됨

<br>
<br>

### 4. Sparse Tensor Network Layers
#### 4.1 Generalized Convolution
- conventional convolution은 discrete convolution이며, input과 output 모두 discrete dense tensor 형태를 가지고 있다.
- 아래는 일반 convolution operation 수식화한 것
![eq3](/assets/images/2021-04-27-sparse-tensor/eq3.png)

- 아래는 sparse convolution operation 수식화한 것
![eq4](/assets/images/2021-04-27-sparse-tensor/eq4.png)

- 차이점은 i 에 있다.
- 위의 수식(4.3)에서 $i$ 는 kernel size 모든 공간을 흝으면서 kernel weight (W)를 input feature value ($f^{in}$)와 곱하면서 $f^{out}$ 을 만들어냄
  - $V^D(K)$는 origin에 center를 둔 D 차원의 hypercube의 offset list를 의미
    - e.g., $V^1(3)$={$-1, 0, 1$}
- 하지만 아래의 sparse convolution 수식 (4.4)은 $i$ 가 $C^{in}$에서 값이 있는 부분들에 대해서만 고려하기에, sparse하게 feature가 update됨
  - $N_D$는 convolution kernel shape을 정의하는 offset들의 set를 의미
  - $N_D$는 arbirarily 정의될 수 있음
    - 이는 dialated convolution, typical hypercubic kernel과 같은 special case들이 적용/포함될 수 있음을 의미함
  - 즉, ![sparsetensor_ind](/assets/images/2021-04-27-sparse-tensor/sparsetensor_ind.png) 는 현재의 point $**u**$을 center로 두고서의 kernel offset과 input coordinates $C^{in}$의 교차되는 부분을 의미

![fig3](/assets/images/2021-04-27-sparse-tensor/fig3.png)

**중요한부분!!!!!**
1. $C^{out}$은 dynamically 생성될 수 있다.(이는 generative task에 대해서는 crucial함)
2. Output coordinates, $C^{out}$ 은 $C^{in}$과 독립적으로 무작위(arbitrarily)하게 정의될 수 있다.
3. Convolution kernel의 shape은 무작위로 $N^D$ 형식으로 정의될 수 있다.

<br>
<br>

---

# Coordinate Manager
- Coordinate Manager: generate a new sparse tensor and findes neighbors among coordinates of non-zero elements
- 일단 새로운 coordinate set를 만들면, coordinate manager는 그 coordinate들과 neighborhood search 결과들을 cache화함
  - coordinate들과 neighborhood search result들은 자주 사용되기에.
- conv layer들, 또는 residual block들은 여러번 불러서 operation시킬 수 있듯이, 위 저장하는것도 반복해서 함
- 다만, same coordinate & same kernel map을 사용해서 매번 재계산하는게 아니라, coordinate manager가 그들의 결과들을 cache화하고, 만약 그 cache dictionary 내에서 같은 operation이 감지된다면, 그 때 저장했던걸 불러와서 재사용한다.
- 아래는 원문
```
A coordinate manager generates a new sparse tensor and finds neighbors among coordinates of nonzero elements. Also, once we create a new set of coordinates, the coordinate manager caches the coordinates and the neighborhood search results these are reused very frequently. For example, in many conventional neural networks, we repeat the same operations in series multiple times such as multiple residual blocks in a ResNet or a DenseNet. Thus, instead of recomputing the same coordinates and same kernel maps, a coordinate manager caches all these results and reuses if it detects the same operation that is in the dictionary is called.
```

## Sparse Tensor Generation
1. Discretization: Unstructured data → Sparse Tensor
  - 원래 data의 continuous coordiate 를 C={$X_i$}$^N_{i=1}$ 라고 한다면, 이를 discretization 시켜줘야함
  - 단순하게, quantization factor $s$ 를 통해 원래의 data, $X$를 나눠주고 flooring해주면 됨 (정수형으로 바꿔주기 위해) 
2. Hash Table: discretized coordinates 를 저장하기 위해
  - Key: D-dimensional 정수형 coordinate
    - 즉,, 그냥 grid라고 생각하면 되려나?
  - Value: 저장된 그 coordinate의 row index가 value가 됨
    - 이거는,, 그 grid안에, 원래의 unstructured data의 index 정보가 들어가있는 것으로 생각하면 될 듯


## Coordinate Key
- Coordinate Key 는 sparse tensor의 coordinate의 정보를 cach화시킨 unordered map을 위한 Hash Key이다.
- 만약 두 개의 sparse tensor가 같은 coordinate manager와 coordinate key를 가지고 있다면, 그 두 개의 sparse tensor의 coordinate는 identical하다는 것이며, 그 둘은 같은 memory space를 공유한다.

## Kernel Map

![fig3](/assets/images/2021-04-27-sparse-tensor/fig4.png)

- 위 그림은 일반적인 convolution(dense convolution)과 sparse convolution과의 비교 그림이다
- Im2col() 에 대한 설명은 다음 링크 참고 [[Im2Col() 참고 링크](https://welcome-to-dewy-world.tistory.com/94)]
  - dense convolution을 수행하기에 앞서, 다차원의 데이터를 2D 행렬로 변환하여 matrix 연산을 할 수 있도록 도와주는 알고리즘 (내적연산, inner-product)
  - 시간복잡도 줄이기 위해 사용
- Sparse Convolution을 위한 Kernel Map은 위의 Im2Col() 함수와 동일한 역할을 수행한다.
  - 한 점 $u$ 주변의 존재하는 coordinate, $N^D(u)∩C^{in}$ 을 찾기 위해 $N(u)$를 정의하는 과정
    - 즉 모든 점에 대해 수행해야하기 때문에, 각각의 데이터 $u$마다 $N(u)$를 정의하는 작업 (iterate)


<br>
<br>

---
# 마무리
- 더욱 자세한 내용은 아래 링크 참고
  - [[Minkowski Engine Doc](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html)]
  - [[3D Semantic Segmentation with Submanifold Sparse Convolutional Neural Networks, CVPR’18](https://arxiv.org/pdf/1711.10275.pdf)]
  - [[4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR’19](https://arxiv.org/pdf/1904.08755.pdf)]
  - [[High-dimensional Convolutional Neural Networks for 3D Perception, Stanford University](https://purl.stanford.edu/fg022dx0979)]
    - [[Chapter 4. Sparse Tensor Networks](https://node1.chrischoy.org/data/publications/thesis/ch4_sparse_tensor_network.pdf)]

---

# 관련 python libray 들
  - [[SparseConvNet](https://github.com/facebookresearch/SparseConvNet)]
    - Pytorch 1.3
    - Cuda 10.0
    - Python 3.3 with conda
    - 상위버전의 pytorch 또는 cuda 11 지원하는지는 잘 모르겠음..
    - 자세한 사항은 위 링크 통해서 확인바람
  - [[spconv](https://github.com/traveller59/spconv/issues)]
    - Pytorch 
    - Cuda 
    - Python 
    - GCC 
    - 자세한 사항은 위 링크 통해서 확인바람
  - [[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)]
    - Pytorch >= 1.7
    - Cuda >= 10.1.243 (please use the same CUDA version used for pytorch)
    - Python >=3.6
    - GCC >= 7.4.0
    - 자세한 사항은 위 링크 통해서 확인바람
  - [[TorchSparse](https://github.com/mit-han-lab/torchsparse)]
    - Pytorch = 1.6.0
    - CUDA 10.2
    - CUDNN 7.6.2
    - 자세한 사항은 위 링크 통해서 확인바람
      - kernel map 구축할 때, GPU사용가능하도록 했기 때문에, Minkowski engine 보다 속도가 빠르다고 함
      - 하지만, MinkowskiEngine에 비해 제한된 함수들만 존재하며, 아직 1.1.0 버전이 나왔을 만큼, 아직 개발 초기단계이기에, API document site 같은 것이 없음
      - 또한, CPU trainig 지원 안 함