## Proposed Efficient Model

### Motivation

To tackle the problem of [], we propose a efficient model.

reduce runtime memory cost. runtime memory is mainly cost by activation map and gradient storage, which directly related to input image size, and model structure.

reduce storage cost. storage is mainly cost by model structure complexity.

reduce time consumption. time consumption is directly related to gflops of model with given size input.

### Model Structure

model structure: shallower, slimer, use depth-wise seperable conv to reduce gflops.

the model consists of []