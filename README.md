# DenseRandomNet
This repository is to provide an implementation of the deep learning tool based on dense random neural network for classification of multi-channel datasets. The related papers are references [1-3]. The implementation is also shared in the author's personal website http://www.yonghuayin.icoc.cc/.

# Requirements
Note that this implementation is based on MATLAB. The tested versions for this implementation include MATLAB R2014a and R2016b.

# Get Started
The following presents an example of using the DenseRandomNet to classify the small NORB data from https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/ [4]. Each sample in this dataset has two 96x96 images. 
## 0. Download Dataset
Download the small NORB dataset from https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/ [5], decompress the files and put them into the "norb_dataset" fold.

## 1. Pre-process the dataset
Open the script "preprocess_norb.m" in MATLAB and run it. This script downsamples the 96x96 images in the norb dataset into 32x32 images. Note that running this script may encounter the "Out of Memory" error in MATLAB if the computer does not have enough memory. Alternatively, for a computer with less memory, use the script "preprocess_norb_small_batch.m" to preprocess the dataset.

## 2.1. Run the main script
Open the script "Use_MCMLDRNN_for_NORB.m" in MATLAB and run it. This script uses the deep learning tool based on dense random neural network, implemented as "MCMLDRNN.m", to learn and classify the downsampled small norb dataset. The trained MCMLDRNN model is stored in ''MCMLDRNN_for_norb.mat''.

The following training and testing accuracies are obtained in a single trial:
TrainingAccuracy = 0.999218106995885
TestingAccuracy = 0.892839506172839

Note that the accuracies can be slightly different in different trials.

## 2.2. Alternatively, run the main script with the whiten preprocessing procedure
Open the script "Use_MCMLDRNN_for_NORB_whiten.m" in MATLAB and run it. This script uses the deep learning tool based on dense random neural network, implemented as "MCMLDRNN.m", to learn and classify the downsampled and whitened small norb dataset. The trained MCMLDRNN model is stored in ''MCMLDRNN_for_norb-whiten.mat''.

The following training and testing accuracies are obtained in a single trial:
TrainingAccuracy = 0.999876543209877
TestingAccuracy = 0.911604938271605

Note that the accuracies can be slightly different in different trials.

# References
[1] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Random Neural Networks. 2016 International Joint Conference on Neural Networks (IJCNN) 2016, pp. 1633–1638.

[2] Yin, Yonghua; Gelenbe, Erol. Deep Learning in Multi-Layer Architectures of Dense Nuclei. In NIPS 2016 workshop: Brains and Bits: Neuroscience Meets Machine Learning (available in https://arxiv.org/abs/1609.07160).

[3] Gelenbe, Erol; Yin, Yonghua. Deep Learning with Dense Random Neural Networks. International Conference on Man-Machine Interactions. Springer, 2017, pp. 3-18.

[4] Yin, Yonghua. "Deep Learning with the Random Neural Network and its Applications." arXiv preprint arXiv:1810.08653 (2018).

[5] Y. LeCun, F.J. Huang, L. Bottou, Learning Methods for Generic Object Recognition with Invariance to Pose and Lighting. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004.

[6] Tang, Jiexiong; Deng, Chenwei; Huang, Guang-Bin. Extreme Learning Machine for Multilayer Perceptron. IEEE Transactions on Neural Networks and Learning Systems, vol. 27, no. 4, pp. 809-821, April 2016.
