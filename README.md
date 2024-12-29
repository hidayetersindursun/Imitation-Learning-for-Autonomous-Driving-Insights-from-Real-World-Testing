# Imitation Learning for Autonomous Driving: Insights from Real-World Testing

This repository contains code and resources for implementing and testing an autonomous driving system, as detailed in *"Imitation Learning for Autonomous Driving: Insights from Real-World Testing"* by Hidayet Ersin Dursun, Yusuf Güven, and Tufan Kumbasar. The paper describes designing deep neural networks (DNNs) that translate raw image inputs into steering commands using an imitation learning framework. The project emphasizes real-time testing and smooth vehicle operation under varying conditions.

## Overview of Architectures

The repository supports different deep learning approaches:

- **PD Controller**: A classical proportional-derivative controller for lane following, discussed as a baseline.
- **CNN**: Predicts steering angles directly from images using transfer learning with ResNet-18.
- **CNN-LSTM**: Combines CNN and LSTM to incorporate temporal dependencies in driving sequences.
- **CNN-NODE**: Integrates Neural Ordinary Differential Equations (NODE) to capture continuous-time driving dynamics.

## Folder Structure

- [cnn](cnn/):
  Implements a Convolutional Neural Network (CNN) approach as outlined in Section III-A of the paper.
  - `createTrainTestData.m`: Prepares datasets for training and testing.
  - `training.m`: Trains the CNN model using ResNet-18.
  - `inference.m`: Performs inference with the trained CNN model.

- [cnn-lstm](cnn-lstm/):
  Contains a CNN-LSTM implementation for handling sequential image data, detailed in Section III-B.
  - `create_network.m`: Prepares the network with sequential input data.
  - `apply_saturation.m`: Adjusts steering angle ranges for training.

- [cnn-node](cnn-node/):
  Provides code for integrating Neural Ordinary Differential Equations (NODE) with CNNs, explained in Section III-C.
  - `trainODE.py`: Trains the CNN-NODE model.
  - `pytorch2onnx.py`: Converts models for deployment.

- [line_following](line_following/):
  Implements a line-following system using a PD controller as a baseline, described in Section II.
  - `data_collection/save_img_and_data.py`: Collects images and control data.
  - `line_follower.py`: Implements line-following logic.

## Key Features

- **Real-Time Testing**: The models are tested on the MIT Racecar platform to assess their performance in navigating tracks with varying conditions.
- **Incremental Design**: The DNNs are developed iteratively to address challenges in real-world driving.
- **Performance Analysis**: Results highlight the advantages of CNN-LSTM and CNN-NODE architectures for smooth and adaptive steering.

## Experimental Results

- **PD Controller**: Effective at low speeds but struggled with sharp turns and lighting variations.
- **CNN**: Improved steering performance but lacked temporal awareness.
- **CNN-LSTM**: Delivered smoother driving by capturing temporal dependencies.
- **CNN-NODE**: Performed comparably to CNN-LSTM with enhanced handling of continuous dynamics.

## Dependencies

- MATLAB for CNN and CNN-LSTM training.
- PyTorch for CNN-NODE implementation.
- NVIDIA TensorRT for optimized real-time inference.

## Citation

If you find this work useful, please cite:

```
@article{dursun2024imitation,
  title={Imitation Learning for Autonomous Driving: Insights from Real-World Testing},
  author={Dursun, Hidayet Ersin and Güven, Yusuf and Kumbasar, Tufan},
  journal={IEEE},
  year={2024}
}
```

For more details, refer to the experimental video at [YouTube](https://www.youtube.com/watch?v=FNNYgU--iaY).
