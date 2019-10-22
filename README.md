
# Preprocessing Audio Samples

## 1 Dataset Overview

<p> From the Kaggle Freesound Audio Tagging dataset, I extracted samples from 10 differents classes. Each one of these classes are a instrument. The list of the classe is the following : </p>
<ul> 
    <li>Saxophone</li>
    <li>Violin_or_fiddle</li>
    <li>Hi-hat</li>
    <li>Snare_drum</li>
    <li>Acoustic_guitar</li> 
    <li>Double_bass</li>
    <li>Cello</li>
    <li>Bass_drum</li> 
    <li>Flute</li>
    <li>Clarinet</li>
</ul>

<p> For each one of theses classes, I picked 30 differents samples of different sizes. Therefore the training dataset has 300 samples. Here is a pie graph of the sample distribtuion for each classes based on sample length : </p>
<img src="class_distribution.png"></img>


### 2 Preprocessed time serie samples

<p> On the X axis we have time (in s). On the y axis we have the amplitude, caracterized by the bit depth. We know tbhat our samples come from a microphone with 44.1kHz and 16 bit depth. That means it is able to pick up 44100 values per second, and each value consist of a 16 bit array. </p>


```python
plot_signals(signals_before_threshold)
plt.show()
```


![png](output_10_0.png)


### 3 Time series with threshold enveloppe

<p> Here we are going to get rid of the parts where the microphone could not record enough data points. In order to do that we are going to use a rolling mean window. For each sample, we are going to look at all the points within a 0.10s window, and if there is at least one point in the window we will compute the mean of all points.Then we will compare that mean with a defined threshold. The threshold I used is 0.0005, I took this value because Librosa resample the wav file using small float values, and after experimenting I got the best results with that value. If the mean of all points is grater than the threshold, I keep the values, otherwise I discard it from the sample. </p>


```python
plot_signals(signals)
plt.show()
```


![png](output_13_0.png)


### 4 Short Time Fourier Transform

<p> First we are going to downsample our audio. The original sample rate is 44.1kHz, but all relevant data points can be represented within a 16kHz rate. I did that by specifying the rate in the librosa.load() function. I then used Numpy fft function to make the Fourier transform. </p>
<p> Applying a Fourier transform to the whole sample at once makes no sence because the frequencies keeps changing over time. Instead of that, we are going to use an overlapping window. We are going to look at all data points within a 25ms windows, make a fast Fourier transform, then move the window to 15ms to the right so it overlap 10ms, make a FFT and so on. We will store every FFT to represent the sample.</p>
<p>We now can use a spectrogram to represent the Short time Fourier transform. The frequency is in the x axis and the magnitude is in the y axis. </p>


```python
plot_fft(fft)
plt.show()
```


![png](output_16_0.png)


### 5 Mel Filter Bank

<p> The Mel scale relates perceived frequency, or pitch, of a pure tone to its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they are at high frequencies. Incorporating this scale makes our features match more closely what humans hear. </p>


```python
plot_fbank(filterbank)
plt.show()
```


![png](output_19_0.png)


### 6 Mel Cepstrum Coefficient

<p> Here we are going to discard the 13 highest frequencies filters to keep lower frequencies only, because change between samples are mostly recognizable in low frequencies </p>


```python
plot_mfccs(mfccs)
plt.show()
```


![png](output_22_0.png)


## 5. Saving the downsampled audio

<p> Here we will downsample the original audio file at a rate of 16000 Hz, then apply a threshold to it, then save the file in a new folder called "clean" </p>


```python
#df.set_index('fname', inplace=True)
for filename in df.index:
    signal,rate = librosa.load('/home/romain/TF Notebooks/Audio Classifier/test_wavfiles/'+filename, sr = 16000)
    mask = threshold_envelope(signal,rate,0.0005)
    wavfile.write(filename='/home/romain/TF Notebooks/Audio Classifier/clean_test/'+filename, rate=rate, data=signal[mask])
```


```python

```
