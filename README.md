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
        │    │   train_00000.png
        │    │   train_00001.png
        │    │   ...
        │
        └─── gt
             │   real_00000.png
             │   real_00001.png
             │   ...
```


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

