## Deep Learning and Brain

This repository is a code sample of the following paper: 
"Correspondence of Deep Neural Networks and the Brain for Visual Textures" Md Nasir Uddin Laskar, Luis G Sanchez Giraldo, and Odelia Schwartz, ArXiv Preprint. Link: https://arxiv.org/abs/1806.02888

The whole project is divided into three main steps:

    step 1: Deep net output
    step 2: Qualitative comparison with the brain data
    step 3: Quantitative comparison with the brain data

Where,

Step 1: Is the process where we send the texture images to the deep net and find the output from each layer.

Step 2: Is the qualitative correspondence of deep neural networks with the brain recording data, on some metrics, as described in the paper.

Step 3: Shows the quantitative correspondence of the deep network with the brain recording data, as described in the paper.


### How to run
Just run the main function. It will invoke the step 2 and you will see a plot in the paper. You can run any part of step 3 also, as the necessary data is provided. To run step 1, you need to have the input images and Caffe installed in your machine. You can also use your own texture dataset of images to find similarity with the brain. For now, we mostly consider texture images as V2 neurons are sensitive to texture type inputs.


### Note
Please see the about.txt and data.txt for more details on the program and necessary data and uml_class_diagram.pdf to get a sense about the class hierarchies.
