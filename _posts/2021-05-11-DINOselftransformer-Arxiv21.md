---
title: "Paper Review 3: [Arxiv 2021] 4DINO: Emerging Properties in Self-Supervised Vision Transformers
"
categories:
  - Paper Review
tags:
  - Arxiv
  - Self-supervised
  - Transformer
  
---

---

# 들어가며

NLP 연구분야에서부터 시작된 transformer 의 성공은 vision 분야에서도 많이 적용되며, 특히 2D와 3D를 넘나들며 여러 분야에 적용 및 발표되고 있다.

하지만, 이러한 노력에도 불구하고, 아직까지는 vision 분야에서 괄목한만한 성능을 보이는 방법들은 없으며, 오히려 다음과 같은 이슈들만 부각되기 시작했다. 
- 많은 양의 메모리 사용량 필요 (multi-head self-attention 에서 attention map 생성할 때, $O(n^2)$ 필요 )
- 구글에서 나온 JTL과 같은 대용량의(imagenet보다 훨씬 더 많은) 데이터셋을 활용힌 학습 요구

이런 이슈들로 인하여, 
1. sparse transformer라는 것들이 발표되기 시작했으며, $O(n^2)$ 를 $O(n log(n))$ 과 같이 complexity를 줄이기 위한 노력들이 이뤄지기 시작했다.
2. 또한, Deit와 같은 논문처럼 대용량의 데이터셋 없이도 할 수 있는 방법론 또한 제기되기 시작했다.

하지만 그 누구도 NLP에서의 BERT, GPT와 같은 논문들처럼 NLP에서의 transformer는 pretraining이 요구되며, 이런 pretraining을 통해 여러 downstream task에서 많은 성공을 이뤄냈다는 점에 집중하지 못했었다. 

하지만 이번 포스트를 통해 소개하는 논문은 이러한 점에 집중했고, Unsupervised learning 방법으로 transformer를 적용, pretraining하는 방법론은 제기하였으며, 기존의 다른 vision transformer 계열들보다 supervised approache와의 gap을 상당히 줄였음을 보여준다.


# Introduction

앞서 말한것처럼, NLP에서의 transformer의 성공은 pretraining에 있다해도 과언이 아닐 것이며, supervised로 학습된 feature보다 self-supervised로 학습된 feature가 더욱 의미있는 정보를 담고 있다는 점이 주목되었고, vision에서 또한 이 점에 주목하여 self-supervised pretraining 을 위한 pretext task를 제안해왔다.

이런 self-supervised/unsupervised 방법론들에 영감을 받아, ViT feature들이 self-supervised pretraining에 미치는 영향에 대해 의문점이 들기 시작하였고, 이와 관련된 실험들을 통해 다음과 같이 흥미로운 사실들을 발견하였다.
```
아래의 발견들은 supervised ViT에서도 발견되지 않았고, 다른 convnet 에서도 발견되지 않은 특징들이다.
```

- Self-supervised ViT features 는 scene layout, 특히, object 경계면에 대한 정보들을 명시적으로 담고 있음을 확인하였고, 이러한 정보는 마지막 block의 self-attention module에서 확인할 수 있었다.
  -  segmentation mask 정보를 얻는다는 것은 대부분의 self-supervised 방법론들에서 보여주는 현상이지만, 
  -  m여기에서 주목해야할 점은 omentum encoder 와 multi-crop augmentation을 적용했을 때에만, K-NN 성능이 높아지는 현상이다.

![eq1](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/fig1.png)

- Self-supervised 로 학습된 ViT features 은 (finetuneing 없이 + linear classifier 도 없고, + data augmentation 없이) K-NN classifier를 통해 분류를 해보았을 때, 성능이 좋게 나옴을 확인할 수 있었다. -> ImageNet에서 78.3%의 Top-1 accuracy를 보였다.
  - 일반적으로 self-supervised/unsupervised 로 학습된 네트워크는 feature extractor로 여겨지며, 제대로된 feature를 뽑았는지를 확인하기 위해서 가장 간단한 linear classifier 또는 K-NN으로 분류 성능을 확인해봄

- ViT 에서 사용되는 image patch 사이즈는 작을 수록 feature의 결과물(품질)을 향상시킬 수 있었다.

