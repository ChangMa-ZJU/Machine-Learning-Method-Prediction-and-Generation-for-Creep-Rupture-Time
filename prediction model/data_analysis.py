import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance



def pre_progressing(file_path):
    original_data = pd.read_excel(file_path)
    target = original_data['RT']
    data = original_data.drop(['RT'],axis=1)
    
    normalize_data = MinMaxScaler().fit_transform(data.values)
    normalize_data = pd.DataFrame(normalize_data,columns=data.columns)
    log_target = np.log(target)
    log_target = pd.DataFrame(log_target)
    normalize_target = MinMaxScaler().fit_transform(log_target)
    normalize_target = pd.DataFrame(normalize_target)
    
    
    return normalize_data, log_target, normalize_target


#棒棒图和包点图
def VIF(X,c1):
    font2 = {'family': 'Times New Roman',
 		     'weight': 'normal',
 		     'size': 20 }
    vif_info = pd.DataFrame()
    vif_info['Column'] = X.columns
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info = vif_info.sort_values('VIF', ascending=False)
    vif_info['Tol'] = 1./vif_info['VIF']
    # vif_info.to_csv('Results/VIF.csv')
    
    fig =plt.figure(figsize=(5,4),facecolor='white',dpi=200)
    ax=fig.add_subplot(111, label="1") 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(axis="y")
    #画柱子
    ax.vlines(x=vif_info['Column'],ymin=0,ymax=vif_info.VIF,color=c1,alpha=1.0,linewidth=2.5);
    #画散点图
    ax.scatter(x=vif_info['Column'],y=vif_info.VIF+20,s=20,color=c1,alpha=1.0)
    
    ax.set(ylabel='VIF')
    ax.set_ylabel('VIF',fontdict={'size':7})
    ax.set_xticks(vif_info['Column'])
    ax.set_xticklabels(vif_info['Column']
                  ,rotation=60
                  ,horizontalalignment='right'
                  ,fontsize=6);
    for i,cty in enumerate(vif_info.VIF):
        ax.text(i,cty+250#注释所有的横纵坐标
                ,round(cty,1) #b保留一位小数
                ,horizontalalignment='center'#相对于我们规定的x和y坐标，文字显示在什么地方
                ,rotation=60
                ,fontsize=6
                )
        
    plt.yticks(fontsize=5)
    ax.spines['right'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    plt.savefig('Figs/VIF.png',dpi=300,bbox_inches = 'tight')
    plt.show()
       
            
#用xgboost特征重要性排序
def get_feature_importance(file_path):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    df = pd.read_csv(file_path)#是归一log后的data
    feature_name = [column for column in df][:-1]
    sample_data = df.values  
    train_dataX = sample_data[:,:-1]
    train_dataY = df[['RT']]
    train_dataY =  MinMaxScaler().fit_transform(train_dataY)
    k = 5 
    kf = KFold(n_splits=k)
    rf = RandomForestRegressor()
    num_round = 100
    rf =XGBRegressor(max_depth=2, learning_rate=1, n_estimators=num_round, 
                   silent=True, objective='binary:logistic')
    rf.fit(train_dataX, train_dataY)
    
    feature_impor_all_data = rf.feature_importances_
    f_importances = np.zeros(33,)
    for train_index, test_index in kf.split(train_dataX):
        trainX, trainY = train_dataX[train_index], train_dataY[train_index]
        testX, testY = train_dataX[test_index], train_dataY[test_index]
        rf.fit(trainX, trainY)
        f_importances +=rf.feature_importances_
    average_importance = f_importances/k
    indices = np.argsort(average_importance)[::-1]#从最后一个元素到第一个元素复制一遍(反向)
    features = []
    feature_importances = []
    for f in range(train_dataX.shape[1]):
        features.append(feature_name[indices[f]])
        feature_importances.append(average_importance[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, feature_name[indices[f]],average_importance[indices[f]]))
    f_im = np.column_stack((features, feature_importances))
    df = pd.DataFrame(f_im, columns=["feature", "impor"])
    df.to_excel("Results/feature_importance.xlsx")
    plot_importance(rf)
    plt.show()


def plot_feature_importance(c):
    df = pd.read_excel("Results/feature_importance.xlsx")
    sample_data = df.values[:, 1:3]   
    y_pos = np.arange(34)
    target = sample_data[:,1]
    impor =((target-target.min())/(target.max()-target.min()))  
    
    fig =plt.figure(figsize=(5,4),facecolor='white',dpi=200)
    ax=fig.add_subplot(111, label="1") 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(axis="y")
    ax.vlines(x=sample_data[:,0],ymin=0,ymax=impor,color=c,alpha=1.0,linewidth=2.5);
    ax.scatter(x=sample_data[:,0],y=impor+0.01,s=20,color=c,alpha=1.0)

    ax.set(ylabel='Importance')
    ax.set_ylabel('Importance',fontdict={'size':10})
    
    ax.set_xticks(df['feature'])
    
    ax.set_xticklabels(df['feature']
                 ,rotation=60
                 ,horizontalalignment='right'
                  ,fontsize=6);

    for i,cty in enumerate(impor):
        ax.text(i,cty+0.05#注释所有的横纵坐标
                ,round(cty,4) 
                ,horizontalalignment='center'
                ,rotation=60
                ,fontsize=6
                )
    ax.spines['right'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    plt.yticks(fontsize=5)
    plt.savefig('Figs/Importance.png',dpi=200,bbox_inches = 'tight')
    plt.show()
    
    
def evaluation_indicator():
    df1 = pd.read_excel("Results/feature_importance.xlsx")
    df2 = pd.read_csv('Results/VIF.csv')
    imp = df1["impor"]
    tol = df2['Tol']
    impor =((imp-imp.min())/(imp.max()-imp.min())) 
    tolerance = ((tol-tol.min())/(tol.max()-tol.min())) 

    features = []
    importances = []
    tolerances = []
    VIM = []    
    for i in range(len(imp)):
        for j in range(len(tol)):
            if df1['feature'][i] == df2['Column'][j]:
                features.append(df1['feature'][i])
                importances.append(impor[i])
                tolerances.append(tolerance[j])
                VIM.append(0.7*impor[i]+0.3*tolerance[j])
    
    
    eva = pd.DataFrame(np.column_stack((features,importances,tolerances,VIM)),columns=['feature', 'importance','tolerance','VIM'])
    eva.to_excel("Results/evaluation.xlsx")
    

def plot_evaluation(c):
    df1 = pd.read_excel("Results/evaluation.xlsx")
    df = df1.sort_values('VIM', ascending=False)
    sample_data = df.values[:, 1:5]   
    target = sample_data[:,-1]
    fig =plt.figure(figsize=(5,4),facecolor='white',dpi=200)
    ax=fig.add_subplot(111, label="1") 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(axis="y")
    ax.vlines(x=sample_data[:,0],ymin=0,ymax=target,color=c,alpha=1.0,linewidth=2.5)
    ax.scatter(x=sample_data[:,0],y=target+0.01,s=20,color=c,alpha=1.0)
    ax.set(ylabel='Evaluation')
    ax.set_ylabel('Evaluation',fontdict={'size':10})
    ax.set_xticks(df['feature'])
    
    ax.set_xticklabels(df['feature']
                 ,rotation=60
                 ,horizontalalignment='right'
                 ,fontsize=6)
    
    for i,cty in enumerate(target):
        ax.text(i,cty+0.04
                ,round(cty,4) 
                ,horizontalalignment='center'
                ,rotation=60
                ,fontsize=6
                )
    plt.yticks(fontsize=5) 
    ax.spines['right'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    plt.savefig('Figs/evaluation.png',dpi=200,bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    
    file_path = "./data/data_logtar.csv"
    
    
    df = pd.read_csv(file_path)
    fea = df[list(df.columns[:-1])]
    tar = df[df.columns[-1]]
    
    cmap=sns.diverging_palette(20, 220, n=200)
    c1=cmap[10]
    c2=cmap[180]
    c3='dimgray'
    
    # Pearson Correlation
    
    # plt.figure(figsize = (50,40),dpi=300,facecolor=None, edgecolor='black')
    # ax=plt.axes()
    # mask = np.zeros_like(fea.corr(), dtype=np.bool_)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(fea.corr(),cmap=sns.diverging_palette(20, 220, n=200),
    #         mask = mask,annot=True,center = 0, annot_kws={"fontsize":15},edgecolors='black')
    # # ax.spines["left"].set_color(c2) # 修改左侧颜色
    # # ax.spines["right"].set_color(c2) # 修改右侧颜色；同第二个y轴的label设置一样，该设置也不起作用
    # # ax.spines["top"].set_color(c2) # 修改上边颜色
    # # ax.spines["bottom"].set_color(c2) # 修改下边颜色
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.savefig('Figs/PC.png',dpi=300)
    # plt.show()
    
    VIF(fea,c1)
    
    # get_feature_importance(file_path)
    plot_feature_importance(c2)
    # evaluation_indicator()
    plot_evaluation(c3)