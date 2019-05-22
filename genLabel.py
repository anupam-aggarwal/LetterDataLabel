
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import time
import os
import shutil


# In[ ]:


files = os.listdir("database")
COL = np.load("COL.npy")


# In[ ]:


db = []
for file in files:
    db = db + pickle.load(open("database/"+file,"rb"))


# In[ ]:


ls = []
for i in range(len(db)):
    ls.append([db[i][0],COL[i],db[i][1]])    


# In[ ]:


np.save("labelledData/dat"+str(int(time.time())),ls)
shutil.rmtree("letters")
os.mkdir("letters")
