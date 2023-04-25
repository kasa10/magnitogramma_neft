#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statistics as st


# def read_data(idx: int):
#     with open(f"train/X/{idx}.pkl", "rb") as f:
#         x = pickle.load(f)
#     with open(f"train/Y/{idx}.pkl", "rb") as f:
#         y_elements, y_defects = pickle.load(f)
#     return x, y_elements, y_defects

def read_data(idx):
    with open(f"train/X/{idx}", "rb") as f:
        x = pickle.load(f)
    with open(f"train/Y/{idx}", "rb") as f:
        y_elements, y_defects = pickle.load(f)
    return x, y_elements, y_defects


#%%
x,y_elements, y_defects =read_data('1001392.pcl')

#%%
plt.figure(figsize=(30,2))
plt.imshow(x, origin='upper')

plt.show()


#%% np.Array to Pandas DF
# importing the modules
import numpy as np
import pandas as pd

df_x = pd.DataFrame(data=x)
df_y_def = pd.DataFrame(data=y_defects)
df_y_el = pd.DataFrame(data=y_elements)

#%%
o=5
m0=[]
m1=[]
m2=[]
m3=[]
m4=[]
for i in range(0,4096):

    if df_y_el[0][i] == 0:
        m0.append(df_x[i].mean())
    if df_y_el[0][i] == 1:
        m1.append(df_x[i].mean())
    if df_y_el[0][i] == 2:
        m2.append(df_x[i].mean())
    if df_y_el[i] == 3:
        m3.append(df_x[i].mean())
    if df_y_el[0][i] == 4:
        m4.append(df_x[i].mean())


c0=st.mean(m0)
c1=st.mean(m1)
c2=st.mean(m2)
try:
    c3=st.mean(m3) #////try
except:
    c3=0

c4=st.mean(m4)



#%%
kf=df_x.mean().mean()
kf0=kf/c0
kf1=kf/c1
kf2=kf/c2
try:
    kf3=kf/c3
except:
    kf3 =0
kf4=kf/c4
print(kf0,kf1,kf2,kf3,kf4)


#%%
print(c0,c1,c2,c3,c4)
#%% ПОПИКСЕЛЬНО СМОТРЕТЬ ВЕТ


#%% Проверка
x,y_elements, y_defects =read_data('4439688')
#%% np.Array to Pandas DF
import numpy as np
import pandas as pd

el=[]
defect=[]

df_x = pd.DataFrame(data=x)
df_y_def = pd.DataFrame(data=y_defects)
df_y_el = pd.DataFrame(data=y_elements)




#%%


def closest_number(number, num1, num2, num3, num4):
    distances = [abs(number - num1), abs(number - num2), abs(number - num3), abs(number - num4)]
    min_distance = min(distances)
    index_of_min = distances.index(min_distance)
    if index_of_min == 0:
        return 0
    elif index_of_min == 1:
        return 1
    elif index_of_min == 2:
        return 2
    else:
        return 4


for i in range(0,4096):
    el.append(closest_number((kf/df_x[i].mean()),kf0,kf1,kf2,kf4))

    # if df_y_el[0][i] == 0:
    #     m0.append(df_x[i].sum())
    # if df_y_el[0][i] == 1:
    #     m1.append(df_x[i].sum())
    # if df_y_el[0][i] == 2:
    #     m2.append(df_x[i].sum())
    # if df_y_el[0][i] == 4:
    #     m4.append(df_x[i].sum())

#%%
el_df = pd.DataFrame(el)


#%%
df_y_el.value_counts()
#%%
el_df.value_counts()


#%%


#%% sclrn
from sklearn.metrics import precision_recall_fscore_support




precision, recall, f1_score, _ = precision_recall_fscore_support(y_elements, np.zeros(len(y_elements)))

print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)


#%%
import os
img1=os.listdir("train/X")
