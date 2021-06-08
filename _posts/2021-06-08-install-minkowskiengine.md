---
title: "Install Minkoski Engine"
categories:
  - Utility
tags:
  - Utility
---


https://github.com/NVIDIA/MinkowskiEngine#anaconda

위 링크를 참고하길 바라며...

</n>

여기에서는 Minkowski 0.5.1 + pytorch 1.7.1 설치방법 정리해본다

</n>

# Install Minkowski Engine

정식 document 사이트에서 말하길,

- Python >= 3.6
- CUDA 10.2 => 오직 Pytorch 1.8.1
- CUDA 11.1 => Pytorch 1.7.1

NVIDIA 3090 GPU를 사용한다면,

- CUDA 11.2를 설치하고, nvidia-graphic driver 또한 460.xx 버전으로 맞추고 시작하길 추천함
- CUDA 11.2를 설치한 뒤, Pytorch 1.7.1을 cuda11.0버전으로 설치

---
* CUDA 11.2 설치 방법

<https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal>

또는

<https://developer.nvidia.com/cuda-toolkit-archive>

에서 원하는 cuda 버전 찾아서 설치하면 됨

---

여기에서보면, cuda 버전이 맞지는 않지만 Minkowski Engine Github Repository issue 탭에서 관련 내용 참고하였음

https://github.com/NVIDIA/MinkowskiEngine/issues/282

---

1. conda 가상환경

```bash
# 가상환경 
conda create -n mk051 python=3.8
conda activate mk051

# Openblas 및 pytorch 설치
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

conda install openblas-devel -c anaconda


# Minkowski Engine 설치
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```

### 주의해야할 점

- openblas를 먼저 설치하고 pytorch를 설치하면, pytorch 설치 안됨(openblas 버전 때문에)

---

</n>

# 설치확인

```python
# pytorch 확인
import torch
torch.__version__
torch.cuda.is_available()

# MinkowskiEngine 설치 확인
import MinkowskiEngine as ME
ME.__version__

# ME Diagnostic
import MinkowskiEngine as ME
ME.print_diagnostics()
```

```print
==========System==========
Linux-5.4.0-74-generic-x86_64-with-glibc2.10
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=18.04
DISTRIB_CODENAME=bionic
DISTRIB_DESCRIPTION="Ubuntu 18.04.5 LTS"
3.8.10 (default, May 19 2021, 18:05:58)
[GCC 7.3.0]
==========Pytorch==========
1.7.1
torch.cuda.is_available(): True
==========NVIDIA-SMI==========
/usr/bin/nvidia-smi
Driver Version 460.80
CUDA Version 11.2
VBIOS Version 94.02.26.08.1C
Image Version G001.0000.03.03
==========NVCC==========
/usr/local/cuda/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
==========CC==========
/usr/bin/c++
c++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

==========MinkowskiEngine==========
0.5.4
MinkowskiEngine compiled with CUDA Support: True
NVCC version MinkowskiEngine is compiled: 11020
CUDART version MinkowskiEngine is compiled: 11020
```

# Minkowski Engine 을 활용한 사례들

- <https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage>

# facebook Hydra 설치

```bash
pip install open3d
pip install hydra-core --upgrade
pip install -U hydra_colorlog
pip install hydra-submitit-launcher --upgrade
```

---

# Migration Guide from v0.4x to 0.5x

- 기존 0.4x 버전보다 0.5x버전이 속도가 빠르다고 함
  - <https://github.com/NVIDIA/MinkowskiEngine/issues/204>
- 따라서, pointcontrast 또는 기타 다른 Minkowski Engine을 사용하는 코드들(v0.4x 사용)에서 v0.5로 설치하고 싶다면,
- 아래의 링크를 통해, 0.4에서 사용되는 코드를 수정해주면 됨
- <https://github.com/NVIDIA/MinkowskiEngine/wiki/Migration-Guide-from-v0.4.x-to-0.5.x>
- <https://github.com/facebookresearch/PointContrast/issues/17>
- <https://github.com/NVIDIA/MinkowskiEngine/issues/292>

다른 0.5 버전 관련 issue 들 from github

- <https://github.com/NVIDIA/MinkowskiEngine/issues/250>