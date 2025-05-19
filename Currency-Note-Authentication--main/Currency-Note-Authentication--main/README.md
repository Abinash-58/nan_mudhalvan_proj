# Currency Note Authentication

## Purpose

The primary aim of this project is to develop a reliable and efficient method for authenticating currency notes using image processing and feature matching. By comparing key features in genuine and test note images, this approach helps in detecting counterfeit notes with minimal manual intervention.

## Technologies Used

- **Python** – Programming language used for implementing the project.
- **OpenCV** – Computer vision library for feature detection, keypoint matching, and image processing.
- **NumPy** – For efficient matrix operations and numerical calculations.
- **Matplotlib** – For visualizing keypoint matches and results.

## Usage

1. **Load genuine and test note images.**  
2. **Extract keypoints and descriptors** using the ORB (Oriented FAST and Rotated BRIEF) algorithm.  
3. **Match the extracted features** using the BFMatcher with Hamming distance for robust comparison.  
4. **Calculate a similarity score** based on the proportion of good matches.  
5. **Display the top feature matches** for visual verification.  
6. **Provide a simple decision** on the authenticity of the test note based on the similarity score.

## Conclusion

This project provides a straightforward yet effective approach to currency note authentication using feature matching techniques. By leveraging ORB's speed and accuracy, it offers a scalable solution for detecting counterfeit notes, potentially aiding in secure financial transactions and fraud prevention.
