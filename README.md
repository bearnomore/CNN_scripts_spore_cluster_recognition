# CNN_scripts_spore_cluster_recognition
### Codes and instructions for using MatLab codes to apply convolutional neural network on Nikon imgs of clustered spores. Each pixel of the image will be classified into "background", "Boundary of the spore cells", or "Inside area of the spore cell".
Whole workflow includes 8 steps:
1. Image collection for boundary Labeling (ImgCollection4boundaryLabeling)
2. Combining mannually and automatically labeled Imgs and split dataset into training and validation sets (CombineLabeledImgs_SplitDataSet)
3. Preprocessing images to centerize each pixel for classification and convert dataset to 4D array format accepted by MatLab ( PreprocessingImgs)
4. Training (Start Training)
5. Cross validation by applying trained network on validation img set (ValidationOnTestImgs)
6. Training and validation by transfer learning from known network structures (TransferLearningAndTesting)
7. Set up metric to test the performance of the network (MetricOfPerformance)
8. Applying network on large images(TestLargeImg)
