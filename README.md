# Handwritten-Digit-Recongnition
A basic python Neural Network model using tensorflow to predict handwritten digits.

## Project Description
This is a basic python neural network project which creates a Sequential neural network using tensorflow library. The model is trained based on predefined mnist dataset available in tensorflow. The model has one input(Flatten) layer, two hidden(Dense) layers and final output(Dense) layer. The model predicts digits with more than 95% accuracy.

## Installation
1. Clone this repo using following command
```
git clone https://github.com/void-1409/Handwritten-Digit-Recongnition
```
2. Change current directory to cloned directory
```
cd Handwritten-Digit-Recognition
```
2. Install required libraries from [requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```
3. To train the model run **train_model.py** file
```
python train_model.py
```
4. A folder named **trained model** will be created in your current directory. Now run **main.py** file to test the images in digits folder using our trained model
```
python main.py
```

`Note:` You can add your own handwritten digits in 28x28 pixel format. Add the image files in **digits** and name them as _digit&lt;number&gt;.png_
