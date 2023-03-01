import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn import datasets
from matplotlib import rc
from sklearn import linear_model
from numpy.linalg import norm
##################################################################################################################################################################
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
##################################################################################################################################################################

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

##################################################################################################################################################################

st.markdown(""" <style> .font_title {
font-size:50px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:50px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:35px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:28px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:26px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext {
font-size:18px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

font_css = """
<style>
button[data-baseweb="columns"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px; font-family: 'times';
}
</style>
"""
def sfmono():
    font = "Times"
    
    return {
        "config" : {
             "title": {'font': font},
             "axis": {
                  "labelFont": font,
                  "titleFont": font
             }
        }
    }

alt.themes.register('sfmono', sfmono)
alt.themes.enable('sfmono')
####################################################################################################################################################################

st.markdown('<p class="font_title">Chapter 4 - Part 1: Loss Functions and Regularizations</p>', unsafe_allow_html=True)

####################################################################################################################################################################
cols = st.columns([2, 2 , 2])
cols[0].image("https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png")
cols[1].image("https://scikit-learn.org/stable/_images/sphx_glr_plot_ridge_path_001.png")
cols[2].image("https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_lars_ic_001.png")
cols = st.columns([6 , 2])
cols[0].markdown('<p class="font_text"> Datasets utilized here include diabetes which has 10 input features and 1 target feature, and fake user-defined data. In case of trying breast cancer dataset, we need to select the input feature (which you can do from select box in the sidebar). Now, we can see how that feature is distributed.</p>', unsafe_allow_html=True)

####################################################################################################################################################################

# Dataset_Name = st.sidebar.selectbox('Select your dataset',('Cancer', 'Fake Linear Sin', 'Fake Nonlinear Sin', 'Fake Sinh', 'Fake Exp', 'Fake Linear', 'Fake Nonlinear'),index = 6)
Dataset_Name = st.sidebar.selectbox('Select your dataset',('Cancer','Fake Linear'),index = 1)

if Dataset_Name == 'Cancer':
    Data = pd.read_csv("diabetes.txt", sep="	")
    Names=list(Data.columns)
    X=Data.iloc[:,0:-1]
    y=Data.iloc[:,-1].to_numpy().reshape(-1,1)
    Feature_Label = st.sidebar.selectbox('Select input feature:',Names[0:-1],index = 0)
    Index=Names.index(Feature_Label)
    X_Train = X.iloc[:,Index].to_numpy().reshape(-1,1)   
else:
    np.random.seed(0)
    Data_Numbers = st.sidebar.slider('Size of fake data:', 20, 200, value=100)
    X_Train = np.random.randint(0, 100,Data_Numbers).reshape(-1,1)
    Noise = np.random.randint(-10, 10,Data_Numbers).reshape(-1,1)
    if Dataset_Name == 'Fake Linear Sin':
        y=5*np.sin(X_Train/10)+0.5*Noise
    elif Dataset_Name == 'Fake Nonlinear Sin':
        y=X_Train*np.sin(X_Train/10)+0.5*Noise
    elif Dataset_Name == 'Fake Sinh':
        y=np.sinh(X_Train)+Noise
    elif Dataset_Name == 'Fake Exp':
        y=2*np.exp(X_Train)+Noise
    elif Dataset_Name == 'Fake Linear':
        y=2*X_Train+Noise
    else:
        y=3.6*X_Train**0.5+Noise
    Data=pd.DataFrame(X_Train,columns=["First"])
    # Polynomial = st.sidebar.selectbox('Consider using polynomial')

Visualization = st.sidebar.checkbox('Visualize input feature?')
if Visualization:
    if Dataset_Name == 'Cancer':
        Fig=alt.Chart(Data).mark_bar().encode(alt.X(Feature_Label+":Q", bin=True),y='count()',).properties(width=800,height=300)
        cols[1].altair_chart(Fig, use_container_width=True)
    else:
        Fig=alt.Chart(Data).mark_bar().encode(alt.X("First:Q", bin=True),y='count()',).properties(width=800,height=300)
        cols[1].altair_chart(Fig, use_container_width=True)
 
cols[0].markdown('<p class="font_text"> Next, we are going to plot the target feature versus the input feature. lets see if we can plot a line that would give us general information about correlation between these features.</p>', unsafe_allow_html=True)
cols[0].markdown('<p class="font_text"> There are other built-in function within sklearn that allows developing a linear regression model considering the impact l1 or l2 norm in loss function such as Lasso, Ridge, and ElasticNet (https://scikit-learn.org/stable/modules/linear_model.html). Lets see the linear visualization for each function varies.</p>', unsafe_allow_html=True)
st.write("[link](https://scikit-learn.org/stable/modules/linear_model.html)")
cols = st.columns([2 , 6])


Slope = st.sidebar.slider('Select a value for the slope',-np.round((np.max(y)-np.min(y))/(np.max(X_Train)-np.min(X_Train)),3)*10, np.round((np.max(y)-np.min(y))/(np.max(X_Train)-np.min(X_Train)),3)*10, value=0.0)
Intercept = st.sidebar.slider('Select a value for the intercept',float(np.min(y)), float(np.max(y)), float((np.min(y)+np.max(y))/2))


Columns = st.columns(4,gap='small')
Lasso = st.sidebar.checkbox('Using Lasso?')
if Lasso:
    Alpha_Lasso = cols[0].slider('Select a value for the L1 penalty constant for Lasso regression', 0.0, 100000.0, value=50.0)
Ridge = st.sidebar.checkbox('Using Ridge?')
if Ridge:
    Alpha_Ridge = cols[0].slider('Select a value for the L2 penalty constant for Ridge regression', 0.0, 100000.0, value=50.0)
Elastic_Net = st.sidebar.checkbox('Using Elastic Net?')
if Elastic_Net:
    Alpha_Elastic_Net = cols[0].slider('Select a value for the constant which multiplies penalty for Elastic Net regression', 0.0, 100000.0, value=50.0)
    L1_Ratio_Elastic_Net = cols[0].slider('Select a value for Elastic Net mixing parameter for penalties', 0.0, 1.0, value=0.5)

Metric = st.sidebar.selectbox('Metric used to calculate loss function: ',('R2','MSE','MAE'), index=0)

Score_Methods = pd.DataFrame(columns = ['Method', 'Score'])

rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)
Fig,ax=plt.subplots(figsize=(12,8))
plt.scatter(X_Train,y,label="Original Data")
if Dataset_Name == 'Cancer':
    plt.xlabel(Feature_Label,fontsize=15)
else:
    plt.xlabel('X',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Y',fontsize=15)
X_Sample=np.linspace(np.min(X_Train),np.max(X_Train),2000).reshape(-1,1)
y_Sample=Slope*X_Train+Intercept
plt.plot(X_Train,y_Sample,color='red',label="Proposed Line")
if Metric =='R2':
    Score_Methods.loc[len(Score_Methods)] = ['Proposed',r2_score(y, y_Sample)]
elif Metric =='MSE':
    Score_Methods.loc[len(Score_Methods)] = ['Proposed',mean_squared_error(y, y_Sample)]
else:
    Score_Methods.loc[len(Score_Methods)] = ['Proposed',mean_absolute_error(y, y_Sample)]
#########################################################################################################
Linear_Obj = linear_model.LinearRegression()
Linear_Obj.fit(X_Train,y)
y_Sample_Linear = Linear_Obj.predict(X_Train)
plt.plot(X_Train,y_Sample_Linear,color='green',label="Linear Regression")
if Metric =='R2':
    Score_Methods.loc[len(Score_Methods)] = ['Linear',r2_score(y, y_Sample_Linear)]
elif Metric =='MSE':
    Score_Methods.loc[len(Score_Methods)] = ['Linear',mean_squared_error(y, y_Sample_Linear)]
else:
    Score_Methods.loc[len(Score_Methods)] = ['Linear',mean_absolute_error(y, y_Sample_Linear)]

#########################################################################################################
if Lasso:
    Lasso_Obj = linear_model.Lasso(alpha=Alpha_Lasso)
    Lasso_Obj.fit(X_Train,y)
    y_Sample_Lasso = Lasso_Obj.predict(X_Train)
    plt.plot(X_Train,y_Sample_Lasso,color='aqua',label="Lasso Regression")
    if Metric =='R2':
        Score_Methods.loc[len(Score_Methods)] = ['Lasso',r2_score(y, y_Sample_Lasso)]
    elif Metric =='MSE':
        Score_Methods.loc[len(Score_Methods)] = ['Lasso',mean_squared_error(y, y_Sample_Lasso)]
    else:
        Score_Methods.loc[len(Score_Methods)] = ['Lasso',mean_absolute_error(y, y_Sample_Lasso)]
#########################################################################################################
if Ridge:
    Ridge_Obj = linear_model.Ridge(alpha=Alpha_Ridge)
    Ridge_Obj.fit(X_Train,y)
    y_Sample_Ridge = Ridge_Obj.predict(X_Train)
    plt.plot(X_Train,y_Sample_Ridge,color='darkorange',label="Ridge Regression")
    if Metric =='R2':
        Score_Methods.loc[len(Score_Methods)] = ['Ridge',r2_score(y, y_Sample_Ridge)]
    elif Metric =='MSE':
        Score_Methods.loc[len(Score_Methods)] = ['Ridge',mean_squared_error(y, y_Sample_Ridge)]
    else:
        Score_Methods.loc[len(Score_Methods)] = ['Ridge',mean_absolute_error(y, y_Sample_Ridge)]
#########################################################################################################
if Elastic_Net:
    Elastic_Net_Obj = linear_model.ElasticNet(alpha=Alpha_Elastic_Net,l1_ratio=L1_Ratio_Elastic_Net,random_state=20)
    Elastic_Net_Obj.fit(X_Train,y)
    y_Sample_Elastic_Net = Elastic_Net_Obj.predict(X_Train)
    plt.plot(X_Train,y_Sample_Elastic_Net,color='magenta',label="ElasticNet Regression")
    if Metric =='R2':
        Score_Methods.loc[len(Score_Methods)] = ['ElasticNet',r2_score(y, y_Sample_Elastic_Net)]
    elif Metric =='MSE':
        Score_Methods.loc[len(Score_Methods)] = ['ElasticNet',mean_squared_error(y, y_Sample_Elastic_Net)]
    else:
        Score_Methods.loc[len(Score_Methods)] = ['ElasticNet',mean_absolute_error(y, y_Sample_Elastic_Net)]

#########################################################################################################
plt.legend(fontsize=15,fancybox=True, framealpha=0.6,bbox_to_anchor=[1, 1])

cols[1].pyplot(Fig)

####################################################################################################################################################################
st.markdown('<p class="font_text"> Now, we want to see how score (or even loss function) varies for each method (linear, ridge, lasso, and elastic-net) based on different metrics. Just click on "Visualize Error Bar Plot Based on the metrics?"</p>', unsafe_allow_html=True)
st.write("[link](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error)")

Error_Bar_Plot = st.sidebar.checkbox('Visualize Error Bar Plot Based on the metrics?')

if Error_Bar_Plot:
    Fig1=alt.Chart(Score_Methods).mark_bar().encode(x=alt.X('Method', axis=alt.Axis(labels=False)),y='Score',color='Method').configure_axis(labelFontSize=20)
    st.altair_chart(Fig1, use_container_width=True)

# Slope_Cont=np.linspace(-10*np.round((np.max(y)-np.min(y))/(np.max(X_Train)-np.min(X_Train)),3),10*np.round((np.max(y)-np.min(y))/(np.max(X_Train)-np.min(X_Train)),3),50)
# Slope_Cont=np.linspace(-np.max(y),np.max(y),50)
# Intercep_Cont=np.linspace(-np.max(y),np.max(y),50)
Slope_Cont=np.linspace(-4,4,50)
Intercep_Cont=np.linspace(-4,4,50)

Intercep_Grid, Slope_Grid = np.meshgrid(Intercep_Cont, Slope_Cont)

Contour = st.sidebar.checkbox('Visualize Loss Error Color Map?')
if Contour:
    st.markdown('<p class="font_subheader">Loss Error Color Map</p>', unsafe_allow_html=True)
    New_Columns = st.columns(3,gap='small')
    Loss_MSE= np.zeros_like(Intercep_Grid)
    Loss_MAE= np.zeros_like(Intercep_Grid)
    Loss_R2 = np.zeros_like(Intercep_Grid)
    for i in range (0,Slope_Cont.shape[0]):
        for j in range (0,Intercep_Cont.shape[0]):
            Loss_MSE[i,j]=mean_squared_error(y,Slope_Cont[i]*X_Train+Intercep_Cont[j])
            Loss_MAE[i,j]=mean_absolute_error(y,Slope_Cont[i]*X_Train+Intercep_Cont[j])
            Loss_R2[i,j] =r2_score(y,Slope_Cont[i]*X_Train+Intercep_Cont[j])
            
    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Loss_MSE, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('MSE',fontsize=24)
    plt.colorbar()
    New_Columns[0].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Loss_MAE, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('MAE',fontsize=24)
    plt.colorbar()
    New_Columns[1].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Loss_R2, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('R2',fontsize=24)
    plt.colorbar()
    New_Columns[2].pyplot(Fig1)

Contour_Lasso = st.sidebar.checkbox('Visualize Lasso Loss Function?')

if Contour_Lasso:
    st.markdown('<p class="font_subheader">Lasso</p>', unsafe_allow_html=True)
    New_Columns2=st.columns(3,gap='small')
    Lasso_MSE    = np.zeros_like(Intercep_Grid)
    Lasso_Penalty= np.zeros_like(Intercep_Grid)
    Lasso_Overall= np.zeros_like(Intercep_Grid)
    for i in range (0,Slope_Cont.shape[0]):
        for j in range (0,Intercep_Cont.shape[0]):
            Lasso_MSE[i,j]=mean_squared_error(y,Slope_Cont[i]*X_Train+Intercep_Cont[j])/2
            Lasso_Penalty[i,j]=Alpha_Lasso*norm(np.array([Slope_Cont[i],Intercep_Cont[j]]).reshape(-1,1),1)
    Lasso_Overall=Lasso_MSE+Lasso_Penalty
    
    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Lasso_MSE, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('MSE',fontsize=24)
    plt.colorbar()
    New_Columns2[0].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Lasso_Penalty, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Lasso L1 Penalty',fontsize=24)
    plt.colorbar()
    New_Columns2[1].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Lasso_Overall, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Lasso Overall Loss',fontsize=24)
    plt.colorbar()
    New_Columns2[2].pyplot(Fig1)
    
Contour_Ridge = st.sidebar.checkbox('Visualize Ridge Loss Function?')

if Contour_Ridge:
    st.markdown('<p class="font_subheader">Ridge</p>', unsafe_allow_html=True)
    New_Columns3=st.columns(3,gap='small')
    Ridge_MSE    = np.zeros_like(Intercep_Grid)
    Ridge_Penalty= np.zeros_like(Intercep_Grid)
    Ridge_Overall= np.zeros_like(Intercep_Grid)
    for i in range (0,Slope_Cont.shape[0]):
        for j in range (0,Intercep_Cont.shape[0]):
            Ridge_MSE[i,j]=mean_squared_error(y,Slope_Cont[i]*X_Train+Intercep_Cont[j])/2
            Ridge_Penalty[i,j]=(Alpha_Ridge/2)*(norm(np.array([Slope_Cont[i],Intercep_Cont[j]]).reshape(-1,1),2)**2)
    Ridge_Overall=Ridge_MSE+Ridge_Penalty
        
    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Ridge_MSE, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('MSE',fontsize=24)
    plt.colorbar()
    New_Columns3[0].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Ridge_Penalty, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Ridge L2 Penalty',fontsize=24)
    plt.colorbar()
    New_Columns3[1].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Lasso_Overall, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('Ridge Overall Loss',fontsize=24)
    plt.colorbar()
    New_Columns3[2].pyplot(Fig1)
 
Contour_Elastic_Net = st.sidebar.checkbox('Visualize ElasticNet Loss Function?')
 
if Contour_Elastic_Net:
    st.markdown('<p class="font_subheader">Elastic Net</p>', unsafe_allow_html=True)
    New_Columns4=st.columns(4,gap='small')
    Elastic_Net_MSE    = np.zeros_like(Intercep_Grid)
    Elastic_Net_L1     = np.zeros_like(Intercep_Grid)
    Elastic_Net_L2     = np.zeros_like(Intercep_Grid)
    Elastic_Net_Overall= np.zeros_like(Intercep_Grid)
    for i in range (0,Slope_Cont.shape[0]):
        for j in range (0,Intercep_Cont.shape[0]):
            Elastic_Net_MSE[i,j]=mean_squared_error(y,Slope_Cont[i]*X_Train+Intercep_Cont[j])/2
            Elastic_Net_L1[i,j]=L1_Ratio_Elastic_Net*norm(np.array([Slope_Cont[i],Intercep_Cont[j]]).reshape(-1,1),1)
            Elastic_Net_L2[i,j]=(Alpha_Elastic_Net/2)*(1-L1_Ratio_Elastic_Net)*(norm(np.array([Slope_Cont[i],Intercep_Cont[j]]).reshape(-1,1),2))**2
    Elastic_Net_Overall=Elastic_Net_MSE+Elastic_Net_L1+Elastic_Net_L2
    
    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Elastic_Net_MSE, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('MSE',fontsize=24)
    plt.colorbar()
    New_Columns4[0].pyplot(Fig1)
    
    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Elastic_Net_L1, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('ElasticNet L1 Penalty',fontsize=24)
    plt.colorbar()
    New_Columns4[1].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Elastic_Net_L2, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('ElasticNet L2 Penalty',fontsize=24)
    plt.colorbar()
    New_Columns4[2].pyplot(Fig1)

    Fig1=plt.figure(figsize=(10,10))
    plt.contourf(Intercep_Cont, Slope_Cont, Elastic_Net_Overall, 1000, cmap='viridis')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title('ElasticNet Overall Loss',fontsize=24)
    plt.colorbar()
    New_Columns4[3].pyplot(Fig1)
    
####################################################################################################################################################################

st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013). </p>', unsafe_allow_html=True)

##################################################################################################################################################################
