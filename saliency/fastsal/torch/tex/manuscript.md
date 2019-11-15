## Proposed Efficient Model

### Motivation

Recently proposed saliency prediction models often adopt classification models that pretrained on large scale classification dataset as backbone. While the performance were boosted, The computational cost grown as well. To evaluate how much saliency task actually utilize the model complexity of large scale classification model, we run PCA on activation maps from last layer of MLNet \cite{cornia2016deep} \footnote{using pretrained weight from https://github.com/marcellacornia/mlnet\label{note}} and it's original VGG Net backbone, and visualize the distribution. The visualization shows that classification task activation maps are more sparse and evenly distributed, while saliency activation maps are relatively densely distributed, inferring that not all of the features are actually contributing for learning the diversity. 

In contrast to the unutilized features, saliency prediction, as a low level visual task, usually runs on devices with constraint computational power, making a requirement of maximum utilization of resources. To tackle this problem of confliction that previous models neglected, we propose a compact and efficient model, to achieve comparable results while cost only minimal computational resources comparing with previous deep models.

\begin{figure}[htp]
    \centering
    \includegraphics[width=0.5\textwidth]{features_mlnet-2.png}
    \caption{Visualization of PCA of last layer activation maps of MLNet and VGG Net backbone}
    \label{fig:1}
\end{figure}

% \includegraphics[width=0.5\textwidth]{features_mlnet.pdf}

### Model Structure

To propose such model, we mainly consider three aspect: (1) size on disk (2) memory requirements and (3) performance. Specifically, we want to minimize the memory and storage requirements and time consumption as possible, while keeping a comparable performance on multiple metrics. 

\begin{table}
\begin{center}
\begin{tabular}{ c c c c c c } 
 \hline
 operator & in & out & k & s & p \\ 
 \hline
 conv2d & 3 & 32 & 3 & 1 & 1 \\
 conv2d & 32 & 32 & 3 & 1 & 1 \\
 maxpooling & - & - & 2 & 2 & 0 \\
 bottleneck & 32 & 16 & 3 & 1 & 1\\
 bottleneck & 16 & 16 & 3 & 1 & 1\\
 maxpooling & - & - & 2 & 2 & 0 \\
 bottleneck & 16 & 24 & 3 & 1 & 1\\
 bottleneck & 24 & 24 & 3 & 1 & 1\\
 bottleneck & 24 & 24 & 3 & 1 & 1\\
 maxpooling & - & - & 2 & 2 & 0 \\
 bottleneck & 24 & 32 & 3 & 1 & 1\\
 bottleneck & 32 & 32 & 3 & 1 & 1\\
 bottleneck & 32 & 32 & 3 & 1 & 1\\
 bottleneck & 32 & 32 & 3 & 1 & 1\\
 bottleneck & 32 & 64 & 3 & 1 & 1\\
 bottleneck & 64 & 64 & 3 & 1 & 1\\
 bottleneck & 64 & 128 & 3 & 1 & 1\\
 bottleneck & 128 & 128 & 3 & 1 & 1\\
 conv2d & 128 & 1 & 1 & 1 & 0 \\
 \hline
\end{tabular}
\end{center}
\label{table:1}
\caption{in: input channels, out: output channels, k: kernel size, s: stride, p: padding}
\end{table}

\noindent\textbf{Size on disk: } To reduce the storage requirement, we start constructing our model from a basic structure which has slightly shallower depth than VGG16 backbone. We also reduce the width of multiple layers. We then adopt depth wise seperable convolution using Bottleneck with expansion layer modules which first introduced in MobileNet V2 \cite{sandler2018mobilenetv2} to replace most of the convolution layers to further reduce the model size.

\noindent\textbf{Memory requirements: } The runtime memory of a model is mainly cost by activation map on forward inference and gradient storage on backward propagation, which directly related to input image size and model structure. With the compact model structure, we perform early pooling to downsize the activation maps at early stage to reduce the overall runtime memory cost.

\noindent\textbf{Performance: } Under a constrained total parameter size, slim and deep neural networks usually outperforms wide and shallow ones \cite{He_2016_CVPR}. Multiple methods were proposed recent years to achieve smaller model while preserving performance, such as network pruning, distillation, etc. Though in which method pruning is most commonly adopted, training from sratch usually outperform the model that pruned to the same size \cite{liu2018rethinking}. So to tradeoff as much performance, we make our model similar depth with VGG16, and reduce the wide for most layers as stated previously. We then train our compact model from scratch to achieve better performance.

Following these concept, we construct our model with 2 3x3 convolution layers, 13 Bottleneck with expansion layer modules, and then followed by 1 1x1 channel reduction convolution layer. We set the expansion rate of each Bottleneck module to 4. The detailed setups is shown in Table 1.