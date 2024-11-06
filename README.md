<div align="center">
  
  # <div align="center"> Cough classification with hybrid CNNs and topology </div>

This repo is a demo of using inspiration and applications from Topological Data Analysis, Audio Processing, and Deep Learning.
It takes different audio sounds and classifies if it is someone coughing or not, so it is a binary classification problem. 
However, a cough sound sounds indistuingishable from other sounds like microphone checking or random background noise for the human ear.
In this repo, the approach has been to produce topological features and Mel Spectograms to create 3D and 2D CNN hybrids to classify if the sound is a cough sound or not!

Specifically, you'll find custom tensor transformations instead of the [Euler Characteristic Transform](https://en.wikipedia.org/wiki/Euler_characteristic). <br>

![Custom Tensor Transformation](https://github.com/jakorostami/cough_classification/blob/main/media/tensorslicing.gif) <br>
<br>

## <div align="center"> Topological Data Analysis </div>
Below is the output from using the Mapper algorithm to use unsupervised machine learning to find clusters of different sounds. 
You'll also find usage of Taken's theorem by creating high-dimensional features for phase space reconstruction.
  
  <p>
    <a target="_blank">
    <img width="70%" src="https://github.com/jakorostami/cough_classification/blob/main/images/mapper_pres.png" alt="Mapper"></a>
    
  </p>

  <b>Taken's Embeddings reduced to PCA Embeddings</b>
  <p>
    <a target="_blank">
      <img width="70%" src="https://github.com/jakorostami/cough_classification/blob/main/images/turkish_woman_speaking_presentation.jpg" alt="Takens and PCA"></a>
  </p>


## <div align="center"> Spectograms and other transforms </div>
Because we are working with audio signals, a natural approach is to produce Mel Spectograms as input for the models.
Another way of also producing features or analysing the signal is to use Zero-Crossing Rates. 
  <p>
    <a target="_blank">
      <img width="70%" src="https://github.com/jakorostami/cough_classification/blob/main/images/melspec.png" alt="Mel Spectogram Cough"></a>
  </p>

<b> Zero-Crossing Rates</b>
  <p>
    <a target="_blank">
      <img width="70%" src="https://github.com/jakorostami/cough_classification/blob/main/images/zcr.png" alt="Mel Spectogram Cough"></a>
  </p>


</div>
