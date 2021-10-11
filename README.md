# birdCLEF2021

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

'''
row_id ''' : ID code for the row.

<mark> site </mark>: Site ID.

<mark> seconds </mark>: the second ending the time window

<mark> audio_id </mark>: ID code for the audio file.

#### train_metadata.csv -
A wide range of metadata is provided for the training data. The most directly relevant fields are:

<mark> primary_label  </mark>: a code for the bird species. You can review detailed information about the bird codes by appending the code to https://ebird.org/species/, such as https://ebird.org/species/amecro for the American Crow.

<mark> recodist  </mark>: the user who provided the recording.

<mark> latitude & longitude  </mark>: coordinates for where the recording was taken. Some bird species may have local call 'dialects,' so you may want to seek geographic diversity in your training data.

<mark> date  </mark>: while some bird calls can be made year round, such as an alarm call, some are restricted to a specific season. You may want to seek temporal diversity in your training data.

<mark> filename  </mark>: the name of the associated audio file.

#### trainsoundscapelabels.csv -

<mark> row_id </mark>: ID code for the row.

<mark> site </mark>: Site ID.

<mark> seconds </mark>: the second ending the time window

<mark> audio_id </mark>: ID code for the audio file.

<mark> birds </mark>: space delimited list of any bird songs present in the 5 second window. The label nocall means that no call occurred.
