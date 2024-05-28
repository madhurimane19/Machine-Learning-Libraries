#!/usr/bin/env python
# coding: utf-8

# # line plot

# In[1]:


import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings


# In[2]:


days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
temperature = [30.7,30,32,31,34,33,33.5,34.3,32.1,32.3,35.9,36,39.6,36.6,35.6]

temp_df = pd.DataFrame({"days" : days , "temperature" : temperature})
sns.lineplot(x = "days", y = "temperature" , data = temp_df)
plt.show()


# In[3]:


data = pd.read_csv(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\tips.csv")
print(data)


# In[4]:


data.shape


# In[5]:


sns.lineplot(x = "total_bill", y = "tip", data = data)


# In[6]:


sns.lineplot(x = "tip", y = "total_bill", data = data)


# In[7]:


sns.lineplot(x = "size", y = "total_bill", data = data)


# In[8]:


sns.lineplot(x = "size", y = "tip", data = data, hue = "sex", style = "sex",
            palette = "hot", dashes = False, markers = ["o","<"], legend= "brief")
plt.title("Line Plot", fontsize = 20)
plt.show()


# In[9]:


plt.figure(figsize = (16,9))
sns.set(style='darkgrid')

sns.lineplot(x = "size", y = "tip", data = data, hue = "sex", style = "sex",
            palette = "hot", dashes = False, markers = ["o","<"], legend= "brief")
plt.title("Line Plot", fontsize = 20)
plt.xlabel("Size", fontsize = 15)
plt.ylabel("Total Bill", fontsize = 15)
plt.show()


# In[10]:


plt.figure(figsize = (16,9))
sns.set(style='darkgrid')

sns.lineplot(x = "size", y = "tip", data = data, hue = "day", style = "day",
            palette = "hot", dashes = False, markers = ["o","<",">","^"], legend= "brief")
plt.title("Line Plot", fontsize = 20)
plt.xlabel("Size", fontsize = 15)
plt.ylabel("Total Bill", fontsize = 15)
plt.show()


# # Histogram/ Distplot

# In[ ]:


sns.distplot(data["size"])
warnings.filterwarnings('ignore')


# In[12]:


sns.distplot(data["tip"])


# In[13]:


sns.distplot(data["total_bill"])


# In[14]:


sns.distplot(data["total_bill"],bins = 100, hist = False)


# In[15]:


sns.distplot(data["total_bill"],bins = 100, kde = False, rug = True)


# In[16]:


plt.figure(figsize = (16,9))

sns.histplot(data["total_bill"],bins = 100, multiple='stack',kde = True, color="g",
            common_norm = True, element = "poly",cbar = True, label = "Total Bill",)
plt.title("Histogram of Total Bill", fontsize = 20)
plt.legend()


# In[17]:


bins = [1,5,10,15,20,25,30,35,40,45,50,55]
plt.figure(figsize = (16,9))

sns.histplot(data["total_bill"],bins = bins, multiple='stack',kde = True, color="g",
            common_norm = True, element = "bars",cbar = True,)
plt.xticks(bins)
plt.title("Histogram of Total Bill", fontsize = 20)
plt.show()


# In[18]:


plt.figure(figsize=(16,9))
sns.set()

sns.distplot(data["total_bill"],
             hist_kws = {'color':'#DC143C', 'edgecolor':'#aaff00','linewidth':4,'linestyle':'--', 'alpha':0.9},
            kde= True, 
             
             fit_kws = {'color':'#8e00ce','linewidth':8,'linestyle':'--', 'alpha':0.9},
             rug = True,
             rug_kws = {'color':'#0426d0', 'edgecolor':'#00dbff','linewidth':3,'linestyle':'--', 'alpha':0.9},
            )


# # Barplot

# In[19]:


sns.barplot(x = 'day' , y = 'total_bill' , hue = 'sex', data = data , )


# In[20]:


order = [ 'Sun', 'Thur', 'Fri','Sat']
sns.barplot(x = 'day' , y = 'total_bill' , hue = 'sex', data = data , order = order)


# In[21]:


hue_order = ['Female','Male']
    
sns.barplot(x = 'day' , y = 'total_bill' , hue = 'sex', data = data , hue_order = hue_order,)   


# In[22]:


sns.barplot(x = 'day' , y = 'total_bill' , hue = 'sex', data = data , estimator = np.max)


# In[23]:


sns.barplot(y = 'day' , x = 'total_bill' , hue = 'sex', data = data , ci = 12, palette = "magma", orient = "h")


# # Scatter Plot

# In[24]:


titanic_df = sns.load_dataset("titanic")
titanic_df


# In[25]:


sns.scatterplot(x ='age', y='fare', data = titanic_df,hue ='sex')


# In[26]:


plt.figure(figsize=(16,9))
sns.scatterplot(x ='age', y='fare', data = titanic_df,hue ='sex', style='who')


# In[27]:


plt.figure(figsize=(16,9))
sns.scatterplot(x ='age', y='fare', data = titanic_df,hue ='sex', style='who',
               size='who', sizes = (100,400))


# In[28]:


plt.figure(figsize=(16,9))
sns.scatterplot(x ='who', y='fare', data = titanic_df,hue ='alive', style='alive',
               size='who', sizes = (100,400))


# In[29]:


plt.figure(figsize=(16,9))
sns.scatterplot(x ='who', y='fare', data = titanic_df,hue ='alive', style='alive',
               size='who', sizes = (100,400), palette= 'inferno_r', alpha= .7)


# In[30]:


plt.figure(figsize=(16,9))
sns.scatterplot(x= 'age', y='fare', data = titanic_df)
sns.lineplot(x= 'age', y='fare', data = titanic_df)
sns.barplot(x= 'age', y='fare', data = titanic_df)


# # Seaborn Heatmap

# In[31]:


arr_2d = np.linspace(1,5,12).reshape(4,3)
arr_2d


# In[32]:


sns.heatmap(arr_2d)


# In[33]:


globalWarming_df=pd.read_csv(r'C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\globalwarming.csv')
globalWarming_df


# In[34]:


globalWarming_df  = globalWarming_df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'], axis=1).set_index('Country Name')
globalWarming_df


# In[35]:


plt.figure(figsize= (16,9))
sns.heatmap(globalWarming_df)


# In[36]:


sns.heatmap(globalWarming_df, vmin =0, vmax= 100)


# In[37]:


plt.figure(figsize=(16,9))
 
sns.heatmap(globalWarming_df, cmap="coolwarm")


# In[38]:


plt.figure(figsize =(16,9))
sns.heatmap(globalWarming_df, vmin =0, vmax= 100, cmap = 'coolwarm', annot = True)


# In[39]:


# robust
 
plt.figure(figsize=(16,9))
 
sns.heatmap(globalWarming_df, robust = True)


# In[40]:


# annot (annotate) parameter
 
plt.figure(figsize=(16,9))
 
sns.heatmap(globalWarming_df, annot = True)


# In[41]:


# annot_kws parameter
 
plt.figure(figsize=(16,9))
 
annot_kws={'fontsize':10, 
           'fontstyle':'italic',  
           'color':"k",
           'alpha':0.6, 
           'rotation':"vertical",
           'verticalalignment':'center',
           'backgroundcolor':'w'}
 
sns.heatmap(globalWarming_df, annot = True, annot_kws= annot_kws)


# In[42]:


plt.figure(figsize=(16,9))
 
sns.heatmap(globalWarming_df, linewidths=4, linecolor="k")


# In[43]:


# change style and format of color bar with cbar_kws parameter
   
plt.figure(figsize=(14,14))
 
cbar_kws = {"orientation":"horizontal", 
            "shrink":1,
            'extend':'min', 
            'extendfrac':0.1, 
            "ticks":np.arange(0,22), 
            "drawedges":True,
           }
 
sns.heatmap(globalWarming_df, cbar_kws=cbar_kws)


# In[44]:


# multiple heatmaps using subplots
 
plt.figure(figsize=(30,10))
 
plt.subplot(1,3,1) # first heatmap
sns.heatmap(globalWarming_df,  cbar=False, linecolor="w", linewidths=1) 
 
plt.subplot(1,3,2) # second heatmap
sns.heatmap(globalWarming_df,  cbar=False, linecolor="k", linewidths=1) 
 
plt.subplot(1,3,3) # third heatmap
sns.heatmap(globalWarming_df,  cbar=False, linecolor="y", linewidths=1) 
 
plt.show()


# In[45]:


# set seaborn heatmap title, x-axis, y-axis label and font size
 
plt.figure(figsize=(16,9))
 
ax = sns.heatmap(globalWarming_df)
 
ax.set(title="Heatmap",
      xlabel="Years",
      ylabel="Country Name",)
 
sns.set(font_scale=2) # set fontsize 2


# In[46]:


# sns heatmap correlation
 
plt.figure(figsize=(16,9))
 
sns.heatmap(globalWarming_df.corr(), annot = True)


# In[47]:


# Upper triangle heatmap
 
plt.figure(figsize=(16,9))
 
corr_mx = globalWarming_df.corr() # correlation matrix
 
matrix = np.tril(corr_mx) # take lower correlation matrix
 
sns.heatmap(corr_mx, mask=matrix)


# In[48]:


# Lower triangle heatmap
 
plt.figure(figsize=(16,9))
 
corr_mx = globalWarming_df.corr() # correlation matrix
 
matrix = np.triu(corr_mx) # take upper correlation matrix
 
sns.heatmap(corr_mx, mask=matrix)


# In[49]:


from  sklearn.datasets import load_breast_cancer

cancer_dataset = load_breast_cancer()


cancer_dataset


# In[50]:


cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))
cancer_df.head(6)


# In[51]:


# sns.pairplot(cancer_df)    # pairplot of all dataset


# In[52]:


sns.pairplot(cancer_df, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness' ])


# In[53]:


sns.pairplot(cancer_df, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness' ], hue = 'target')


# In[54]:


sns.pairplot(cancer_df, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness'], hue ='target', palette='Dark2')


# In[55]:


sns.pairplot(cancer_df, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness'], hue ='target', kind = 'reg')


# In[ ]:


sns.pairplot(cancer_df, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness'], hue ='target', diag_kind = 'hist')


# In[ ]:


sns.pairplot(cancer_df, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness'], hue ='target', markers = ['*', "<"])


# In[ ]:




