# Sanskrit Letter Data Label
This repository contains Python code for a web UI to help in labelling images containing sanskrit letters. This is done to create dataset for for building a Deep Learning model to predict these letters.

# Dependencies
1. Python 3.6
2. OpenCV
3. Flask framework

# Usage
---> place image to be segmented in page folder and delete any images in letters folder and files in database folder if they contains any.

---> use python FinalSegment.py ./page/<image_name> to create BW.npy and COL.npy
---> python UI.py
---> go to web browser and type 127.0.0.1:5000/words/process and label the images
---> close the browser, then use command python genLabel.py, labelled data will be stored in labelledData folder.

Data will be a numpy file containing numpy array,
Each array element will be a list of 3 Elements viz, BW image, COL image, Label.
