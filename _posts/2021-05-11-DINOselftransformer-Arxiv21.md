---
title: "Paper Review 4: [Arxiv 2021] DINO: Emerging Properties in Self-Supervised Vision Transformers
"
categories:
  - Paper Review
tags:
  - Arxiv
  - Self-supervised
  - Transformer
use_math: true
---


---

# 들어가며

NLP 연구분야에서부터 시작된 transformer 의 성공은 vision 분야에서도 많이 적용되며, 특히 2D와 3D를 넘나들며 여러 분야에 적용 및 발표되고 있다.

하지만, 이러한 노력에도 불구하고, 아직까지는 vision 분야에서 괄목한만한 성능을 보이는 방법들은 없으며, 오히려 다음과 같은 이슈들만 부각되기 시작했다.

- 많은 양의 메모리 사용량 필요 (multi-head self-attention 에서 attention map 생성할 때, $$O(n^2)$$ 필요 )
- 구글에서 나온 JTL과 같은 대용량의(imagenet보다 훨씬 더 많은) 데이터셋을 활용힌 학습 요구

이런 이슈들로 인하여,

1. sparse transformer라는 것들이 발표되기 시작했으며, $$O(n^2)$$ 를 $$O(n log(n))$$ 과 같이 complexity를 줄이기 위한 노력들이 이뤄지기 시작했다.
2. 또한, DeiT와 같은 논문처럼 대용량의 데이터셋 없이도 할 수 있는 방법론 또한 제기되기 시작했다.

하지만 그 누구도 NLP에서의 BERT, GPT와 같은 논문들처럼 NLP에서의 transformer는 pretraining이 요구되며, 이런 pretraining을 통해 여러 downstream task에서 많은 성공을 이뤄냈다는 점에 집중하지 못했었다.

하지만 이번 포스트를 통해 소개하는 논문은 이러한 점에 집중했고, Unsupervised learning 방법으로 transformer를 적용, pretraining하는 방법론은 제기하였으며, 기존의 다른 vision transformer 계열들보다 supervised approache와의 gap을 상당히 줄였음을 보여준다.


# Introduction

앞서 말한것처럼, NLP에서의 transformer의 성공은 pretraining에 있다해도 과언이 아닐 것이며, supervised로 학습된 feature보다 self-supervised로 학습된 feature가 더욱 의미있는 정보를 담고 있다는 점이 주목되었고, vision에서 또한 이 점에 주목하여 self-supervised pretraining 을 위한 pretext task를 제안해왔다.

이런 self-supervised/unsupervised 방법론들에 영감을 받아, ViT feature들이 self-supervised pretraining에 미치는 영향에 대해 의문점이 들기 시작하였고, 이와 관련된 실험들을 통해 다음과 같이 흥미로운 사실들을 발견하였다.

**아래의 발견들은 supervised ViT에서도 발견되지 않았고, 다른 convnet 에서도 발견되지 않은 특징들이다.**

- Self-supervised ViT features 는 scene layout, 특히, object 경계면에 대한 정보들을 명시적으로 담고 있음을 확인하였고, 이러한 정보는 마지막 block의 self-attention module에서 확인할 수 있었다.
  - segmentation mask 정보를 얻는다는 것은 대부분의 self-supervised 방법론들에서 보여주는 현상이지만,
  - 여기에서 주목해야할 점은 omentum encoder 와 multi-crop augmentation을 적용했을 때에만, K-NN 성능이 높아지는 현상이다.

![fig1](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/fig1.png)

- Self-supervised 로 학습된 ViT features 은 (finetuneing 없이 + linear classifier 도 없고, + data augmentation 없이) K-NN classifier를 통해 분류를 해보았을 때, 성능이 좋게 나옴을 확인할 수 있었다. -> ImageNet에서 78.3%의 Top-1 accuracy를 보였다.
  - 일반적으로 self-supervised/unsupervised 로 학습된 네트워크는 feature extractor로 여겨지며, 제대로된 feature를 뽑았는지를 확인하기 위해서 가장 간단한 linear classifier 또는 K-NN으로 분류 성능을 확인해봄

- ViT 에서 사용되는 image patch 사이즈는 작을 수록 feature의 결과물(품질)을 향상시킬 수 있었다.

## DINO

- teacher 와 student network로 이루어진 구조이며, 각각 encoder-decoder 구조로 이루어져 있음
  - student는 teacher 의 output을 cross-entropy를 활용하여 예측하려하고,
  - Teacher의 output을 Centering하고 sharpening 하는 것만으로도 collapse를 피할 수 있다고 함
    - 물론 predictor, advanced normalization, contrastive loss 도 도움을 주지만, 그 효과는 미미
  - 또한, internal normalizations 도 필요 없이, model architecture 의 수정도 필요 없이, ViT 또는 Convolution Network에 모두 적용가능하기에 flexible하다고 할 수 있다

- 기존 knowledge distillation과의 차이점
  - 기존 knowledge distillation은 teacher는 이미 powerful 한 상태로 freeze 시키고나서, teacher 의 결과를 student에게 넘겨주는 방식
  - 여기에서는 teacher 도 학습 중에 계속 파라미터가 바뀌게 된다.
  - 매 epoch마다 한 번, parameter update를 시켜주는데, "Exponential moving average"를 활용해서, student의 parameter가 일부 전파되는 식으로 update를 시켜준다.
  - 즉, online-distillation (codistillation) 방식을 채택하여 진행한다.

- Network 구조는 다음과 같다.

![fig2](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/fig2.png)

### SSL with Knowledge Distillation

- Knowledge Distillation(KD)을 활용한 알고리즘은 다음 그림 algorithm 에 설명되어 있다.

![algorithm](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/algo1.png)

- 우선 Knowledge Distillation은 student network, $$g_{\theta_s}$$를 teacher network, $$g_{\theta_t}$$ 를 통해서 학습시키는 방법론이며, student network의 output의 확률분포는 다음과 같이 표현될 수 있다.

![eq1](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/eq1.png)

- $$\tau_*$$는 temperature 파라미터이며, sharpeness를 조절하는 역할을 한다.

- 일반적으로 KD 에서는 student의 확률분포가 teacher 의 확률분포를 따르도록, 즉, cross-entropy 를 통해서 학습시키지만(식(2)),

![eq2](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/eq2.png)

- 여기에서는 self-supervised 방법이기에, 식(2)를 다음과 같이 변형을 한다.

![eq3](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/eq3.png)

- input $x$가 있을 때, 여러 개의 view 를 생성한다.
- augmentation을 통해 변형시킨 view들을 사용하는데, 여기에서는 2 개의 global views, $x^g_1$, $x^g_2$, 를 만들고, 여러 개의 local views를 생성한다. (Multi-crop 활용)
  - Global Views: 224-by-224 크기의 영상 -> 원래의 original 영상에서 50% 이상 크기
  - Local Views: 96-by-96 크기의 영상 -> 원래의 original 영상에서 50% 이하 크기
- 모든 crop된 view 들은 student network에 들어가고, 오직 global view들만 teacher network로 들어가서 각각의 output들을 비교하며 $\theta_s$를 학습시키게 된다.
- 서로 다른 view들의 비교를 통해, **Local-to-Global** correspondence 를 확습시킬 수 있다.

#### **Teacher Network**

- 일반적인 Knowledge distillation과 다른 점은 앞서 언급했듯이, 강력한 성능의 teacher network가 존재하지 않고, online-distillation(codistillation)으로 student network와 같이 학습되는 teacher network가 존재한다.
- 하지만, 실제 $\theta_t$가 backprop 통해서 학습되는 것이 아닌, 아래의 수식처럼, exponential moving average(EMA) 방식을 통해 teacher network의 파라미터가 업데이트된다.

![EMA](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/EMA.png)

- 여기에서 $\lambda$ 는 학습동안 cosine schedule을 따르며, 0.998에서 1까지 변화한다.

#### **Network Architecture**

- ViT에 DINO를 적용한 것이고, student 와 teacher 의 구조가 같은 형태이기 때문에, predictor 사용 안함
- ViT ($f$) 끝에, projection head (MLP구조, $h$)를 추가해서 projection head 결과($g=h◦f$)를 학습에 활용하고, downstream task에서는 $f$의 결과를 활용함
- 또한, ViT에서는 batch normalization 이 없기에, 여기에서도 BN-free 구조를 가지고 있음(even projection head에도 BN 없음)

#### **Avoiding Collapse**

- model collapse를 방지하기 위해서, 다른 self-supervised 논문들은 contrastive loss, clustering constraints, predictor, 또는 batch normalization을 적용한다.
- 여기에서는 오로지 teacher output에 centering, sharpening을 적용함으로써, model collapse를 방지한다.
- centering은 아래의 식에서처럼, bias term인 c 를 output에 더해줌으로써 행해진다.

![centering](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/centering.png)

- c 는 EMA할 때, update되며, batch size에 따라 조절된다. update 식은 다음과 같다.

![eq4](/assets/images/2021-05-11-DINOselftransformer-Arxiv21/eq4.png)

- m은 rate parameter이며($m>0$), B는 batch size를 뜻한다.

- Sharpening은 teacher softmax normalization에서의 $\tau_t$를 통해 얻어질 수 있다.
