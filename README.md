# birdCLEF2021

The task involves identifying 397 different bird species vocalizations in audio clips.<br />
The solution employed involves:
1. Training an ensemble of CNN model and ResNets. 
2. Augmented audio data by adding white and pink noise and overlaying other bird vocalizations at a subdued amplitude. 
3. Converting audio clips into 5-second snippets of melspectrograms. 
4. Using adam optimizer and cosine annealing scheduler to train the model. 

The model was used for sound event detection on testsoundscapes, to identify bird vocalization at every 5-second interval in test audio files, giving an F1-score of 0.6.
 
### Data :
https://www.kaggle.com/c/birdclef-2021

### Files

#### trainshortaudio -
The bulk of the training data consists of short recordings of individual bird calls generously uploaded by users of xenocanto.org. These files have been downsampled to 32 kHz where applicable to match the test set audio and converted to the ogg format. The training data should have nearly all relevant files; we expect there is no benefit to looking for more on xenocanto.org.

#### train_soundscapes -
Audio files that are quite comparable to the test set. They are all roughly ten minutes long and in the ogg format. The test set also has soundscapes from the two recording locations represented here.

#### testsoundscapes -
When you submit a notebook, the testsoundscapes directory will be populated with approximately 80 recordings to be used for scoring. These will be roughly 10 minutes long and in ogg audio format. The file names include the date the recording was taken, which can be especially useful for identifying migratory birds.

This folder also contains text files with the name and approximate coordinates of the recording location plus a csv with the set of dates the test set soundscapes were recorded.

#### test.csv -
Only the first three rows are available for download; the full test.csv is in the hidden test set.
``` row_id ``` : ID code for the row.

``` site  ``` : Site ID.

``` seconds  ```: the second ending the time window

```audio_id ```: ID code for the audio file.

#### train_metadata.csv -
A wide range of metadata is provided for the training data. The most directly relevant fields are:
``` primary_label  ```: a code for the bird species. You can review detailed information about the bird codes by appending the code to https://ebird.org/species/, such as https://ebird.org/species/amecro for the American Crow.

``` recodist  ``` : the user who provided the recording.

``` latitude & longitude ``` : coordinates for where the recording was taken. Some bird species may have local call 'dialects,' so you may want to seek geographic diversity in your training data.

``` date ``` : while some bird calls can be made year round, such as an alarm call, some are restricted to a specific season. You may want to seek temporal diversity in your training data.

``` filename ``` : the name of the associated audio file.

#### trainsoundscapelabels.csv -

``` row_id ```: ID code for the row.

``` site ``` : Site ID.

``` seconds ``` : the second ending the time window

``` audio_id ``` : ID code for the audio file.

``` birds ``` : space delimited list of any bird songs present in the 5 second window. The label nocall means that no call occurred.

### Setting up conda env

``` conda create --name birdclef python=3.7 -y ``` <br />
``` conda activate birdclef ``` <br />
``` conda install pytorch torchvision cudatoolkit=10.2 -c pytorch ``` <br />
``` conda colorednoise librosa sklearn  ``` <br />

### References
https://www.kaggle.com/stefankahl/birdclef2021-sample-submission <br />
https://www.kaggle.com/hidehisaarai1213/pytorch-training-birdclef2021-starter <br />
https://www.kaggle.com/hidehisaarai1213/birdclef2021-infer-between-chunk <br />
https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection


