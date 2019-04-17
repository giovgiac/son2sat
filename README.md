# Guided Sonar-To-Satellite Translation

## Dataset

Image files for training should be placed in the correct structure
inside the datasets folder. The structure should be like this:

```
datasets
│   aracati.py    
│
└───aracati
    │
    └───test
    │   │
    │   └─── input
    │   │    │   test_00000.png
    │   │    │   test_00001.png
    │   │    │   ...
    │   │
    │   └─── gt
    │        │   real_00000.png
    │        │   real_00001.png
    │        │   ...
    |
    └───train
    │   │
    │   └─── input
    │   │    │   train_00000.png
    │   │    │   train_00001.png
    │   │    │   ...
    │   │
    │   └─── gt
    │        │   real_00000.png
    │        │   real_00001.png
    │        │   ...
    |
    └───validation
        │
        └─── input
        │    │   valid_00000.png
        │    │   valid_00001.png
        │    │   ...
        │
        └─── gt
             │   real_00000.png
             │   real_00001.png
             │   ...
```

Below is a download link for the ARACATI 2017 dataset used in the papers, if you use it, please cite one of the following (or both):
* [Conference Paper](https://ieeexplore.ieee.org/abstract/document/8614099)
* [Journal]()

[Download ARACATI 2017 Dataset](https://drive.google.com/file/d/1R-tcU67vl_jnnhqvbuE3gMJ9LfOREUpm/view?usp=sharing)

If you'd like to use the same VGG weights we did, here's the download link for them:

[Download VGG16 Weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)

## Executing

To train a new neural network, just run the following comamnd:
```bash
 $ python main.py
```

You can configure execution with the following commands:
* **--mode**: Choose one from: evaluate, restore or train.
* **--name**: Name of the folder to store the files of the experiment.
* **--weights_file**: Weights for the VGG16 network.
* **--batch_size**: Batch size to use for the network.
* **--gen_filters**: Parameter that scales the size of the network.
* **--input_channels**: Number of channels in the input images.
* **--image_height**: Height of the images to run through the network.
* **--image_width**: Width of the images to run through the network.
* **--max_to_keep**: Maximum number of checkpoints to keep.
* **--num_epochs**: Number of epochs to execute the network for.
* **--output_channels**: Number of channels in the output images.
* **--learning_rate**: Initial learning rate for the optimizer.

## Visualizing

After you've executed the script to train the neural network, it will print the tensorboard command you need to use to
visualize the network as it's training. Example below:
```bash
$ tensorboard --logdir='/home/nautec/Documents/Projects/son2sat/executions/2019-04-16_22:08/summary'
TensorBoard 1.13.1 at http://127.0.0.1:6006 (Press CTRL+C to quit)
```
