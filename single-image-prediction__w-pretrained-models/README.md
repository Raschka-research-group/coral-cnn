# Single Image Predictions

Due to the large file sizes of the model, the pre-trained models are not available in this GitHub repository but are shared via Google Drive instead. Below are instructions for using these pre-trained models to make predictions on .jpg files.



## 1) Downloading the models

Visit the Google Drive link via your web browser at [https://drive.google.com/drive/folders/168ijUQyvGLhHoQUQMlFS2fVt2p5ZV2bD?usp=sharing](https://drive.google.com/drive/folders/168ijUQyvGLhHoQUQMlFS2fVt2p5ZV2bD?usp=sharing). There should be 3 subfolders,

- morph2 (725 Mb)
- cacd (744 Mb)
- afad (730 Mb)

which contain the models for each respective dataset.

## 2) Making predictions

There are three Python script files in this directory

- ce.py (regular cross entropy)
- ordinal.py (Niu et al. ordinal regression)
- coral.py (CORAL ordinal regression)

The models can be executed
via the following shell (terminal) commands:

### Cross entropy-based ResNet-34 classifier

```
python ce.py --dataset afad \
--image_path example-images/afad/18_years__948-0.jpg \
--state_dict_path ../afad/afad-ce__seed1/best_model.pt              
```


Output: 

```           
Class probabilities: tensor([[6.9999e-03, 9.3442e-03, 2.6857e-02, 7.9845e-02, 1.9216e-01, 4.3945e-01,
         1.1403e-01, 5.1849e-02, 2.2000e-02, 1.7506e-02, 1.9361e-02, 1.0340e-02,
         4.2141e-03, 1.8779e-03, 1.6173e-03, 5.1615e-08, 8.1854e-04, 4.1623e-06,
         4.9016e-04, 4.3236e-04, 2.8226e-04, 2.7673e-04, 5.1856e-08, 2.2791e-04,
         2.1825e-08, 1.8387e-05]])
Predicted class label: 5
Predicted age in years: 20
```

Note the class labels in the training sets start at 0, which is why the true age (`Predicted age`) is larger than the predicted label (`Predicted class label`).



### Niu et al. Ordinal Regression w. ResNet-34 

```
python ordinal.py --dataset afad \
--image_path example-images/afad/18_years__948-0.jpg \
--state_dict_path ../afad/afad-ordinal__seed1/best_model.pt    
```

Output:

```
Class probabilities: tensor([[9.9470e-01, 9.8194e-01, 9.6384e-01, 9.1268e-01, 7.6474e-01, 5.9748e-01,
         4.3897e-01, 2.9948e-01, 2.0706e-01, 1.3781e-01, 8.4818e-02, 4.4908e-02,
         3.0308e-02, 1.9418e-02, 1.2340e-02, 1.3033e-02, 9.0761e-03, 8.7384e-03,
         5.9033e-03, 3.4582e-03, 2.3147e-03, 6.9869e-04, 6.6803e-04, 1.1985e-04,
         1.3445e-04]])
Predicted class label: 6
Predicted age in years: 21
```


### CORAL Ordinal Regression w. ResNet-34 

```
python coral.py --dataset afad \
--image_path example-images/afad/18_years__948-0.jpg \
--state_dict_path ../afad/afad-coral__seed1/best_model.pt
```

Output:

```
Class probabilities: tensor([[7.9409e-01, 6.6242e-01, 5.0478e-01, 2.5930e-01, 6.9571e-02, 2.1320e-02,
         7.6356e-03, 2.5884e-03, 1.1306e-03, 3.9728e-04, 1.4751e-04, 6.7989e-05,
         3.1884e-05, 1.5259e-05, 6.9953e-06, 6.9953e-06, 3.0060e-06, 2.9677e-06,
         1.4319e-06, 6.5273e-07, 2.5818e-07, 8.4094e-08, 8.4094e-08, 2.2461e-08,
         2.2461e-08]])
Predicted class label: 3
Predicted age in years: 18
```

Note that if you would like to try out CACD or MORPH2 images, you need to change all three arguments accordingly, for example

```
python coral.py --dataset cacd \
--image_path example-images/cacd/41_Jason_Statham_0003.jpg \
--state_dict_path ../cacd/cacd-coral__seed2/best_model.pt
```

Output:

```
Class probabilities: tensor([[1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
         1.0000e+00, 1.0000e+00, 1.0000e+00, 9.9999e-01, 9.9998e-01, 9.9996e-01,
         9.9991e-01, 9.9981e-01, 9.9961e-01, 9.9918e-01, 9.9837e-01, 9.9683e-01,
         9.9390e-01, 9.8826e-01, 9.7705e-01, 9.5824e-01, 9.2423e-01, 8.6348e-01,
         7.7465e-01, 6.3956e-01, 4.8542e-01, 3.2803e-01, 2.0232e-01, 1.2037e-01,
         6.7961e-02, 3.7153e-02, 1.9279e-02, 1.0134e-02, 5.1062e-03, 2.5095e-03,
         1.2001e-03, 5.4535e-04, 2.3731e-04, 9.9439e-05, 4.1659e-05, 1.6209e-05,
         6.1432e-06, 2.3529e-06, 8.2844e-07, 2.3559e-07, 6.8404e-08, 1.2028e-08]])
Predicted class label: 26
Predicted age in years: 40
```