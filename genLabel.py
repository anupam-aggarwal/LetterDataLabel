
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import time
import os
import shutil


# In[ ]:


files = os.listdir("database")
files = sorted(files)
print(files)
COL = np.load("COL.npy")
BW = np.load("BW.npy")
# In[ ]:


db = []
for file in files:
    db = db + pickle.load(open("database/"+file,"rb"))


# In[ ]:


ls = []
for i in range(len(db)):
    ls.append([BW[i],COL[i],db[i][1]])    


# In[ ]:


np.save("labelledData/dat"+str(int(time.time())),ls)
shutil.rmtree("letters")
os.mkdir("letters")
shutil.rmtree("database")
os.mkdir("database")
os.remove("BW.npy")
os.remove("COL.npy")
