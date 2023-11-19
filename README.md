# Dance to Music

## Introduction
Welcome to DanceToMusic, an innovative project that blends my interest in computer vison, audio digital signal processing and generative AI to create novel music based on a sequence of human poses. My goal with this project is to gain experience applying concepts from these different fields. In its current state, this model takes in a 5 sec video, and returns a 5 second piece of novel audio that corresponds with the movement’s of the dancer in the video. 

## Motivation
Inspired by the recent 'text-to-image' (DALL-E, Midjourney, Stable Diffusion) and 'text-to-music' models (MusicLM, MusicGen), this project aims to develop 'Dance-to-Music'. Dance to Music is a technology that could have applicability in the world of social media or content creation, by allowing users to add custom music to their dance videos. Currently this model generates music based off the movements of a single subject in the video, however I can see it being easily extended to generate music conditioned on the movement of multiple subjects in a video. Now imagine if one were to install a camera inside a nightclub, and instead of there being a DJ, trust DanceToMusic to generate music in real-time, that is corresponding the movements of everyone in the club, what would that sound like?

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [EnCodec Model](#encodec-model)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Next Steps](#next-steps)
- [Results](#results)
- [Credits](#credits)
- [License](#license)


## Features
- Video input analysis for human pose estimation.
- Generation of music through a sequence-to-sequence transformer model.
- Novel audio generation from predicted latent space representations.

## Technologies Used
- [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) for 3D human pose estimation
- [Meta's EnCodec model](https://huggingface.co/docs/transformers/main/model_doc/encodec#transformers.EncodecModel) used to create audio encodings from .wav files, as well as a decoder that reconstructs audio from the encoded representation
- Python, PyTorch

## Installation
One can start playing with this project by first cloning the repository, building the dataset, and then running the training script. 
```bash

# Clone the repository
git clone https://github.com/azeezabdikarim/DanceToMusic.git

# Navigate to the repository
cd DanceToMusic

# It is recommended to create a virtual environment for the project to avoid conflicts with other packages and versions
python -m venv venv

# Activate the virtual environment
# On Windows, use `.\venv\Scripts\activate`
# On Unix or MacOS, use `source venv/bin/activate`
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt


# Build the video dataset, extracting 5 second clips at 24fps
# FYI: The video downloads might take a while depending on your internet speeds. Also, MediaPipe uses the CPU to calculate human pose estimates, so that will take a decent amount of time. It takes me between 1-2 hours to build the complete dataset of ~3200 clips on a MacBook Pro with an M2 Max chip
python ml/data/building_tools/build_complete_dataset.py --output_path ml/data/samples/ --input_csv ml/data/youtube_links/youtube_links_test.csv  --max_seq_len 5 --fps 24

```
## Usage

## EnCodec Model
The EnCodec model is a state-of-the-art high fidelity audio encoder developed by the FAIR team at Meta. The model follows an encoder-decoder type of architecture with the latent space represented by a residual vector quantized latent space. The model is trained with perceptual loss where the generated samples were judged by a discriminator, to reduce artifacts and generate high quality samples. 

Adding a vector quantized codebook to an autoencoder architecture, such as the EnCodec model, enables a discrete representation of the latent space, which can be advantageous for handling complex data distributions and improving the robustness of the model against small input variations. This discretization also facilitates the learning of a more meaningful and structured latent space, which can be beneficial for downstream tasks like generation or retrieval.

When the codebook is extended to a Residual Vector Quantized (RVQ) Codebook, the model gains the ability to capture finer details by representing residuals or differences between the input and the current reconstruction at each stage. This layered approach of quantization allows the model to iteratively refine its output, leading to a more precise and higher fidelity reconstruction than what could be achieved with a standard Vector Quantized Codebook.

![EnCodec Model Architecture](ml/assets/encodec_architecture.jpg "EnCodec Model Architecture")

## Dataset
The dataset used in this projects is a personally aggregated set of dance videos paired with their corresponding music. Each dance clip is processed using MediaPipe's 3D human pose estimation model. Only one human is selected per video clip, for which the model generates a set of 33x3 key points per frame. Videos are standardized in the training set to last 5 sec at 24fps, resulting in a sequence of 120 frames per sample. The final dimensions for an individual sample are therefore 120x33x3. 

![Human Pose Estimate of a Single Frame](ml/assets/women_dancing.jpg "Human Pose Estimate Sample")

For each dancing clip, I extracted the audio as a .wav file. I use the '24khz' encoder of the EnCodec audio encoder, to the map the 5 second audio clip to a lower dimensionality latent space. For a 5 second clip, the EnCodec model produces a latent vector of 2x377, which I flattened to 1x754, and use as the target the samples of our dataset. 

## Model Architecture 
From a high-level point of view, the goal of this project is to produce a sequence-to-sequence model capable of mapping a sequence of 33x3 human pose key points to a latent vector, which represents a sequence of codebook values. One of the advantages of using the audio codes generated by the EnCodec model for our target is the fact that the latent space is represented by a sequence of discrete values from a codebook, meaning the model only needs to learn to predict values of a fixed vocabulary. In the case of the '24khz' EnCodec model, the length of the codebook is 1024. In comparison to the text based tasks where the vocabulary can be thousands of distinct values, 1024 values is a relatively small vocabulary, which increases the chances for success.

In the context of our training dataset of 5 sec clips filmed at 24fps with an audio sample rate of 24khz, our goals is to map a vector sequence of 120x33x3 to a latent audio code sequence of 1x754. Due to the layered approach of quantization in a RVQ codebook, our sequence-to-sequence model most reconstruct the audio codes autoregressively. 

## Training
The PoseToMusic transformer is trained to encode the sequence of human pose estimates, and with this context the decoder is tasked autoregressively predicts the sequence of codebook values that matches the target, which is EnCodec's encoding of the audio file. Training is conducted using a 85%-15% training and validation split which accounts for (~2000 training and ~350 validation samples).

The autoregressive nature of the model means that the decoder generates new values for the sequence based on the previous values it has already generated. This type of training can take a while to converge due to the fact that the decoder can compound mistakes, which is why I implemented teacher-forcing 50% of the time. With teacher forcing, previous correct values from the target are provided to the decoder, which in turn helps it converge faster. 

Given the fact that codebook values are discrete, and not sequentially related to one another (eg. Codebook index 1 is not necessarily closer to index 2 than it is to index 55), each codebook value is treated like a class label. For each newly generated token the decoder outputs a softmax vector equal in length to the length of the codebook. The argmax of this vector is taken, and the result is filled in as the next token of the sequence. 

Each generated sequence of audio codes is compared to the target and scored using NLLoss. We train to minimize NLLoss, and as it declines, DanceToMusic learns to generate more realistic music. 

Training stops once the validation loss platues, as I want to avoid overfitting to the training data.

## Results
Currently the results are poor. An example is provided bellow. The audio generated is slightly better than random noise as we can hear some musicality, however it doesn't come close to obviously matching the movement of the dancer.

![My Generated Audio](ml/assets/generated_audio.wav)

<figure style="float: left; margin-right: 20px;">
  <video src="assets/sample_output_video_original.mp4" width="400" height="300" controls="controls"></video>
  <figcaption>Original Audio</figcaption>
</figure>

<figure style="float: left;">
  <video src="assets/sample_output_video.mp4" width="400" height="300" controls="controls"></video>
  <figcaption>Generated Audio</figcaption>
</figure>

<div style="clear: both;"></div>


## Next Steps
I have a few theories for why the model struggles to produce suitable music to match the subjects movements, with the majority focused on how to improve the quality of the training data.

### Training Data
#### Music
The audio samples present in the training dataset is incredibly diverse. The audio that the subjects are dancing to includes salsa, kpop, rock, russian rock, pop music, hip hop music, etc. Beyond the stylistic differences in the music, some of the music has vocals, while other samples don't have vocals. Due to this diversity and the limited number of samples < 3,000 it's possible that the model doesn't have enough samples to truly learn the commonalities across the different styles.
#### Pose Estimates
For some samples, there are errors in the calculation of the human pose estimates. This results in inconstancies of the location of the key points across the various frames. These inconsistencies are likely confusing the model.

### Improve the Training Data
I am investigating a few options for how to improve the quality of the training data. 
One simple solution is to record myself dancing to music for an extended period of time. This will allow me to select simpler music, ensure that the camera remains steady throughout the video, and keep the style/skill level of the dancing consistent.
Another idea is to simplify the music in my current dataset. This might look like processing all the audio with a high-pass filter which would result in audio more oriented around beat and percussion. This is likely to remove the vocals from all the tracks, leaving in the lower frequencies, which may make audio samples more similar. Furthermore, it's usually the lower frequencies of a track that a person's body is responding to, so this filtering technique might help the model focus on producing the most "danceable" part of a track. 

## Credits

## License
