# Face to Sona live "AI" mapping

This app provide a visualiser and training helpers for AI models mapping a face to a sona while preserving the pose, at video speed.

It support two backend network architectures:

- GNR: GANs'n roses: heavier, but can disentangle style from pose, which means it is capable of representing different styles with the same model (for instance: varying fur color)
- CUT: Contrastive Unpaired Translation: lighter, but no style parameter

## Visualiser

Main features:

- input: single image, video file, webcam
- run multiple models in parallel for comparison
- live visualisation, save video, and fake webcam output

### Fake webcam output requirements

Feature provided by pyvirtualcam library.

On windows, you just need OBS installed.

On linux you need v4l2loopback. Run:

    sudo modprobe v4l2loopback devices=1

to enable


## Training your models

### Overview

The training GUI covers all the steps needed to go from a few shot pictures of your sona face to a trained GNR or CUT model:

- train a stable diffusion Lora from your few shots
- generate thousands of images of your sona using the Lora
- train CUT or GNR model

### Detailed instructions

#### Step 1: Lora training

The CUT/GNR training requires thousands of images with some level of consistency, which can't be realistically produced by a human.

So the idea is to train a "patch" to stable diffusion to draw a specific sona from only a few images.

Note: if you are looking for a model rendering a generic sona, or a lot of different sonas which you can produce using a SD model, you can skip this step and go directly to step 2.

The first step is to gather a handful of head crop pictures of the target sona.
What matters more than number is:

- style and features consistency
- variety of poses (front, side, mouth open/closed...)

Create a folder containing your PNG image crops of the sona head.

Finally get a base SD model. You can get the original SD, or one model specifically trained for your style/sona kind of interest. It must not be a Lora.


Then start the training GUI on the first tab.

You need to set at least:

- Base SD model: path to the ".safetensors" base model file
- few shot images dir: the directory with your sona head pictures
- class word: a single english word best describing your sona (ex: wolf, tiger, robot, ...)
- trigger word: use "cpc", this is a word that will trigger SD to draw the sona.

Then hit "run" and go grab something to drink. On a decent GPU this will take less than an hour.

The resulting lora will go in "experiments/lora".


#### Step 2: generating many images

We will need thousands of pictures to train CUT or GNR.

This steps will use a SD model (+ optionally a trained lora from step 1) to do that.

To create images with varying poses and mouth opening, it uses a prompt template system.

You can tinker it at will, but you need to at least adjust the base prompt.

Copy sd-prompts.json to any .json file and open it in a text editor.

Modify the "prompt" value to at least match your sona category or species.
If using lora the prompt should contain your trigger world (like "cpc")

When you're ready open the training GUI to tab 2 "SD many shots".

You need to fill:

- base SD model: .safetensors file with base model
- LoRA: optional Lora (trained at step 1)
- Prompt template: your JSON file

Give it a try with a low number of images to verify everything is in order.

Hit "RUN" and go grab a sandwitch as it may take a few hours.

The image files are written in "experiments/sd-many...".


#### Step 3: Actual model training

Chose between GNR and CUT. GNR is heavier (longer to train, lower framerate) but gives better results and have style control.

Enter in "Domain B image folder" the folder with generated images in step 2.
At least 1000 images are recommended.

The models have many parameters that impact rendering quality and run cost.

If you run out of video RAM when initialising training, you can lower the batch size parameter.

Hit run and go take some much needed vacations as this will take a few days.


Note: the GNR checkpoints are big because they contain more things than strictly needed. Use the utils/gnr-strip.py utility script to keep only the required part if you want to share your model.


