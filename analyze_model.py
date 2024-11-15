from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,precision_score,recall_score,f1_score,accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#数据预处理
data=pd.read_excel(r"F:\consumption_prediction\dataset.xlsx")
data['gender']=data['gender'].replace({1:0,2:1})
data['av_incoming'].fillna(data['av_incoming'].mean(),inplace=True)
data['edu'].fillna(data['edu'].mean(),inplace=True)

features = ['age', 'edu', 'homesize', 'av_incoming', 'urban_index', 'gender']

def display_feature_importance(model, features, title):
    print(f'\n{title}特征重要性')
    
    # 提取特征重要性并创建 DataFrame
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()  # 反转Y轴，以便显示最重要的特征在顶部
    plt.show()

def train_classifier_model(X,y,param_grid=None,test_size=0.2,random_state=42):
    if param_grid is None:
        param_grid={
            'n_estimators':[100],
            'max_depth':[10],
            'min_samples_split':[2],
            'min_samples_leaf':[1]
        }
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    grid_search=GridSearchCV(RandomForestClassifier(random_state=random_state),param_grid,cv=5,scoring='neg_mean_squared_error')
    grid_search.fit(X_train,y_train)
    return grid_search.best_estimator_,X_test,y_test

def train_regressor_model(X,y,param_grid=None,test_size=0.2,random_state=42):
    if param_grid is None:
        param_grid={
            'n_estimators':[100],
            'max_depth':[10],
            'min_samples_split':[2],
            'min_samples_leaf':[1]
        }
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)

    grid_search=GridSearchCV(RandomForestRegressor(random_state=random_state),param_grid,cv=5,scoring='neg_mean_squared_error')
    #初始化对象
    grid_search.fit(X_train,y_train)
    #启动超参数最优组合搜素

    return grid_search.best_estimator_,X_test,y_test

def analyze_province_data(data,features):
    #嵌套函数无法被外部函数调用，保证代码的隔离性
    def prepare_data_for_province(province_data,features):
        X=province_data[features]
        y_soy=province_data['soy']
        y_meat=province_data['meat']
        return X,y_soy,y_meat

    #按省份分析特征重要性
    provinces=data['province'].unique()#对重复性数据去重

    for province in provinces:
        
        province_data=data[(data['province']==province)]#布尔索引实现数据筛选
        
        if len(province_data)<10:
            print(f"{province}数据量过少，跳过分析")
            continue
        #if...continue如果满足if条件则跳出当前循环，进行下一个循环
        X,y_soy,y_meat=prepare_data_for_province(province_data,features)

        best_soy_model,X_test_soy,y_test_soy=train_regressor_model(X,y_soy)
        best_meat_model,X_test_meat,y_test_meat=train_regressor_model(X,y_meat)

        display_feature_importance(best_soy_model,features,f"{province} Feature Importance for Soy Consumption")
        display_feature_importance(best_meat_model,features,f"{province} Feature Importance for Meat Consumption")
    
#analyze_province_data(data, features)

def whether_consume(data,features):
    def consume_data_process(data):
        y_soy=np.where(data['soy']>0,1,0)
        y_meat=np.where(data['meat']>0,1,0)
        return y_soy,y_meat
    
    X=data[features]
    y_soy,y_meat=consume_data_process(data)

    best_soy_model,X_test_soy,y_test_soy=train_classifier_model(X,y_soy)
    best_meat_model,X_test_meat,y_test_meat=train_classifier_model(X,y_meat)

    predictions_soy=best_soy_model.predict(X_test_soy)
    predictions_meat=best_meat_model.predict(X_test_meat)

    print("豆类消费分类准确率:",accuracy_score(y_test_soy,predictions_soy))
    print("肉类消费分类准确率:",accuracy_score(y_test_meat,predictions_meat))

    display_feature_importance(best_soy_model,features,'features impact on soy consuming whether or not')
    display_feature_importance(best_meat_model,features,'features impact on meat consuming whether or not')

#whether_consume(data,features)

def consumption_factor(data,features):
    X=data[features]
    y_soy=data['soy']
    y_meat=data['meat']

    best_soy_model,X_test_soy,y_test_soy=train_regressor_model(X,y_soy)
    best_meat_model,X_test_meat,y_test_meat=train_regressor_model(X,y_meat)
    
    predictions_soy=best_soy_model.predict(X_test_soy)
    predictions_meat=best_meat_model.predict(X_test_meat)

    print("大豆消费量回归MSE",mean_squared_error(y_test_soy,predictions_soy))
    print("肉类消费量回归MSE",mean_squared_error(y_test_meat,predictions_meat))
    
    display_feature_importance(best_soy_model,features,'features impact on soy consumption')
    display_feature_importance(best_meat_model,features,'features impact on meat consumption')

#consumption_factor(data,features)