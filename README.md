# Leveraging Deep Q-Network Agents with Dynamic Routing Mechanisms in Convolutional Neural Networks for Enhanced and Reliable Classification of Alzheimer’s Disease from MRI Scans

This work is part of conference paper _17th International Conference on Agents and Artificial Intelligence 2025_ Porto, Portugal. 

Link to paper: https://www.scitepress.org/PublicationsDetail.aspx?ID=2bQwqzLUWkI=&t=1

**Abstract**

With limited data and complex image structures, accurate classification of medical images remains a significant challenge in AI-assisted diagnostics. This study presents a hybrid CNN model with a capsule network layer and dynamic routing mechanism, enhanced with a Deep Q network (DQN) agent, for MRI image classification in Alzheimer’s disease detection. The approach combines a capsule network that captures complex spatial patterns with dynamic routing, improving model adaptability. The DQN agent manages the weights and optimizes learning by interacting with the evolving environment. Experiments conducted on popular MRI datasets show that the model outperforms traditional methods, significantly improving classification accuracy and reducing misclassification rates. These results suggest that the approach has great potential for clinical applications, contributing to the accuracy and reliability of automated diagnostic systems.

Over the past decade, convolutional neural networks
(CNNs) have transformed fields like computer vision,
image classification, object detection, and segmentation. Their key strength lies in their
ability to replicate learned features across the spatial
dimensions of an image. By using spatial reduction
layers, CNNs can extract local translation-invariant
features, ensuring that object translations in the input
do not affect the activation of high-level neurons.
This is achieved through techniques likemax-pooling,
which transfer features across layers. However, this
translation invariance comes at the cost of losing precise
object location encoding—a long-standing issue
that researchers have sought to address through various
techniques.

Magnetic resonance imaging (MRI) is a more
commonly used test because it offers better image
quality and higher resolution, allowing changes in
brain structures to be seen more accurately. In addition,
MRI does not use ionizing radiation, making
this test safer for the patient. MRI is more effective
in detecting neurodegenerative changes, such as
neuronal atrophy and changes in brain tissues, which
are characteristic of Alzheimer’s disease. MRI allows
accurate assessment of cerebral structures, such
as the hippocampus and glial cortex, often affected by
Alzheimer’s disease.

## References

If you use my code, please cite my work.

Podolszanska, J. (2025). Leveraging Deep Q-Network Agents with Dynamic Routing Mechanisms in Convolutional Neural Networks for Enhanced and Reliable Classification of Alzheimer’s Disease from MRI Scans.
