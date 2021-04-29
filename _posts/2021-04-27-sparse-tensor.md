---
title: "Sparse Tensor Networks
"
categories:
  - Paper Review

---

---

# 들어가며,
- spatially dense 한 language 또는 image와는 다르게, 3D point cloud 또는 higher-dimensional data(e.g., data statistics)들은 공간상에서 data 분포도가 매우 sparse 함
- 효율적인 learning을 위해서는 sparse representation을 어떻게 잘 활용하느냔가 관건임
- 이를 해결하기 위해 spatially sparse data를 활용해서 spatially-sparse convolutional neural network들이 개발되고, 이러한 network들은 spatially sparse tensor를 도입하고, 이에 맞는 sparse tensor를 홀용한 activation 함수들도 제안됨
- 이러한 network들은 **<u>Sparse Tensor Network</u>**이라 불리며, network에 들어가는 input 을 포함한 모든 것들이 sparse tensor로 구성되어 있음
---

<br>
<br>
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
<br>

### 2. Tensor Stride
image stride 개념과 같음

<br>
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
  - 즉, ![sparsetensor_ind](/assets/images/2021-04-27-sparse-tensor/sparsetensor_ind.png) 는 현재의 point **$u$**을 center로 두고서의 kernel offset과 input coordinates $C^{in}$의 교차되는 부분을 의미

![fig3](/assets/images/2021-04-27-sparse-tensor/fig3.png)

**중요한부분!!!!!**
1. $C^{out}$은 dynamically 생성될 수 있다.(이는 generative task에 대해서는 crucial함)
2. Output coordinates, $C^{out}$ 은 $C^{in}$과 독립적으로 무작위(arbitrarily)하게 정의될 수 있다.
3. Convolution kernel의 shape은 무작위로 $N^D$ 형식으로 정의될 수 있다.

<br>
<br>

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
