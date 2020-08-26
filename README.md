## Deep Learning and Brain

This repository is a code sample of the following Journal of Vision paper: 
"Deep Neural Networks Capture Texture Sensitivity in V2" by Md Nasir Uddin Laskar, Luis G Sanchez Giraldo, and Odelia Schwartz. [Paper Link](https://jov.arvojournals.org/article.aspx?articleid=2770349&resultClick=1).

Deep convolutional neural networks (CNNs) trained on visual objects have shown intriguing ability to predict some response properties of visual cortical neurons. This project finds the compatibility of CNN outputs with brain V2 data and pinpoints exactly where V2-like sensitivity occurs in the deep networks. The whole project is divided into three main steps:

    step 1: Find the deep network output from each layer
    step 2: Qualitative comparison of CNN with the brain data
    step 3: Quantitative comparison with the brain data

Where,

Step 1: Is the process where we send the inputs, which are texture images, to the deep network and find the outputs from each layer.

Step 2: Is the qualitative correspondence of deep neural networks with the brain recording data, on some metrics, as described in the paper.

Step 3: Shows the quantitative correspondence of the deep network with the brain recording data, as described in the paper.


### How to run
Just run the main function. It will invoke the step 2 and you will see a plot in the paper. You can run any part of step 3 also, as the necessary data is provided. To run step 1, you need to have the input images and Caffe installed in your machine. You can also use your own texture dataset of images to find similarity with the brain. For now, we mostly consider texture images as V2 neurons are sensitive to texture type inputs.


### Note
Please see the `about.txt` and `data.txt` for more details on the code and necessary data, and `uml_class_diagram.pdf` to get a sense on the class hierarchies.
