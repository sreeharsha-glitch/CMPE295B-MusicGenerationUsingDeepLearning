# CMPE295B-MusicGenerationUsingDeepLearning
## _This repo is the work done by the following team members for the masters project at San Jose State University_ 

### Team members:
- Shanmukh Shiva Sai Ganesh Krishna Boddu - shanmukhshivasai.boddu@sjsu.edu
- Satya Sai Roopa Sree Chiluvuri - satyasairoopasree.chiluvuri@sjsu.edu
- Sai Sree Harsha Chimbili - saisreeharsha.chimbili@sjsu.edu
- Sai Sri Harshini Kosuri - saisriharshini.kosuri@sjsu.edu

### Advisor:
- Prof. Mahima Agumbe Suresh

### Published in
- 

## Project Architecture
The architecture of the project can be broadly divided as the steps shown in fig. 1.
The system architecture can be delineated into three primary components, each serving a distinct role:
A. Model Training in Colab
This initial phase involves training machine learning models within Google Colab. The trained models are then saved in the form of .h5 files. These files encapsulate the learned patterns parameters of the models, allowing for future use within the web application.
B. User Interaction with Web Application Colab
The second phase involves user interaction within the web application. Users are presented with an intuitive interface where they can engage with the system. This interaction includes selecting one of three available algorithms from a dropdown menu – specifically, Long Short-Term Memory (LSTM), Transformers, or Generative Adversarial Networks (GANs). Additionally, users have the option to upload a .mid file as input, providing a musical sequence to serve as a basis for further generation.
C. Algorithmic Music Generation
Following the user's selection, the web application leverages the chosen algorithm and the corresponding pre-trained .h5 file. If the user opts for LSTM, for instance, the application uses the LSTM model's .h5 file to generate music sequences that align with the characteristics of the uploaded .mid file. This process is repeated for the other algorithms as per the user's choice. In essence, the application dynamically tailors its music generation approach based on the user's algorithmic preference and input musical data. This three-fold architecture ensures a cohesive and flexible system, seamlessly integrating model training, user interaction, and algorithmic music generation within a web-based environment.
Below fig. 2 represents the user interface screen

<image>
  
Dataset:
For this research we are working/using Lakh Piano Dataset. This dataset was curated by Music and AI lab at Research centre for IT Innovation, Academia Sinica. The entire dataset contains 174,154 multitrack piano rolls. There are multiple versions of this dataset like lpd, lpd-cleansed, lpd-5 etc., For this research we are using lpd-5 version, which is a combination of various instruments like guitar, drums, piano to make the model multiinstrument predictor. A subset of lpd-5 which is called lpdcleansed is also used which comprises of 21,425 tracks. The lpd-cleansed version contains the files in the MIDI format which is easy to handle by the models and also avoid the preprocessing steps.
The below figure shows the midi file visualization using music21 package.

<image>
  
The below figure shows the midi file visualization using
music21 package.
<image>

## Project description

## Project poster

## Tech Used

## Steps to reproduce
