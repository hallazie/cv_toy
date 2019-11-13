## Proposed Efficient Model

### Motivation

Most of previous models are constructed with a classification-task pretrained model (VGG16, ResNet, etc.). Classification models are too complex for low level task such as saliency prediction. We visualize the distribution of some final layer activation maps of saliency models and classification models with VGG16 as backbone. The classification task activation maps are more sparse and evenly distributed, while saliency activation maps are relatively densely distributed.

To tackle the problem of [resource constraint conditions] which previous models neglected, we propose a compact and efficient model, named xxx. xxx archieves comparable results, and cost only minimal computational resources comparing with previous deep models.

To propose such model, we mainly consider three aspect stated previously: (1) memory requirements and size on disk, (2) number of math-ematical operations, and (3) speed. We want to optimize the memory requirements (run time RAM cost) and model size (storage cost) as possible, while keeping a comparable performance. Generally speaking, the performance of a model is mainly related to it's model complexity, namely gflops.

reduce runtime memory cost. runtime memory is mainly cost by activation map and gradient storage, which directly related to input image size, and model structure.

reduce storage cost. storage is mainly cost by model structure complexity.

reduce time consumption. time consumption is directly related to gflops of model with given size input.

### Model Structure

To achieve smaller model, multiple tricks are proposed recent years, such as pruning, distilling, etc. As stated in [rethinking pruning], training from sratch usually outperform pruning to the same size. So we design a compact model and train it from scratch. The principle of designing is to get as smaller as possible while preserve comparable performance. 

model structure: slimer, use depth-wise seperable conv to reduce gflops.

the model consists of 2 3x3 conv layers, 13 inverted residual modules followed by 1 1x1 channel reduction conv layer. The model use depth seperable conv layers to reduce connection between two conv layers, while keep the channel width.

