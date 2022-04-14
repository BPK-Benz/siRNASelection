import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.utils import resample


def DataDescription(data):
   stat = pd.DataFrame(data.groupby(['Class']).agg(['count']).iloc[:,-1])
   stat.columns = ['Count']
   stat['Percentage'] = round(stat['Count']/stat['Count'].sum()*100,3)
   print(stat) 

def Normalization(data): # Normalized data
    RS = preprocessing.RobustScaler()
    scaled_df = pd.DataFrame(RS.fit_transform(data.iloc[:,:-1]) , columns= data.iloc[:,:-1].columns, index=data.index)
    scaled_df['Target'] = data['Class']
    return scaled_df

def ModelFitPrediction (X_train, X_test, y_train, y_test, model ): # For evaluating model performance
    print('Number of test set: ', X_test.shape[0])

    # Model_Name = 'RandomForestClassifier'
    
    model.fit(X_train, y_train) # step 2: fit
    y_pred=model.predict(X_test) # step 3: predict
    print('Accuracy score:', model.score(X_test, y_test).round(3)) # step 4: accuracy score for classification r2 for regression
    classes=list(set(data['Class']))
    print(metrics.classification_report(y_test, y_pred, target_names=classes))
    y_pred_prob = model.predict_proba(X_test)[::,1]

    # show classification result
    result = pd.concat([y_test.reset_index(),pd.DataFrame([y_pred,y_pred_prob]).T],axis=1)
    result.columns = ['Patient','Class','Predicted','Predicted_Prob']
    print(result)
    return model

def FoldCrossValidation(scaled_df, model): # For describing model generalization
    print('Performance of classification model before feature selection')
    cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
    X=scaled_df.iloc[:,:-1]
    print('Number of all features ', X.shape)
    y=scaled_df['Target']
    scores = cross_val_score(model, X,y, scoring='accuracy',cv=cv,n_jobs=-1)
    print('Scores: ', np.round(scores,4))
    print('Accuracy: %.3f (%.3f)'% (np.mean(scores),np.std(scores)))

    # Feature selection by using Random Forest
    importance = model.feature_importances_
    importance_score = pd.DataFrame([X.columns, importance]).T
    importance_score.columns = ['miRNA','Score']
    importance_score = importance_score.sort_values(by='Score', ascending=False)
    miRNA1 = importance_score[importance_score['Score'] > 0]['miRNA']

    print("\n",'*'*30)
    print('Performance of classification model after feature selection')
    cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
    X=scaled_df.loc[:,miRNA1]
    print('Number of Selected features ', X.shape)
    y=scaled_df['Target']
    scores = cross_val_score(model, X,y, scoring='accuracy',cv=cv,n_jobs=-1)
    print('Scores: ', np.round(scores,4))
    print('Accuracy: %.3f (%.3f)'% (np.mean(scores),np.std(scores)))

    return miRNA1


base = os.getcwd()

# 1. Read Raw data
df = pd.read_csv(base+'merge set 2and4.csv').T
df = df.rename(columns=df.iloc[0]).iloc[1:,:]

label_df = pd.read_csv(base+'Factors set2and4 (3 groups).csv').iloc[:,:3]
label_df['Type'] = label_df['Type'].str.replace(' ','')
label_df['Class'] = label_df['Type']
label_df = label_df.set_index('Sample Name')

rawData = pd.concat([df,label_df['Class']], axis=1).fillna(0)
data = rawData

# 2. Show data description
DataDescription(data=data)
scaled_df = Normalization(data=data)
# 3. Train and Test model 
# Train and Test spliting with 0.67 and 0.33 respectively
X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:,:-1], scaled_df['Target'], 
                                                    test_size=0.33) # , random_state=7
model=RandomForestClassifier() # step 1: choose model/estimator for classification
model = ModelFitPrediction (X_train, X_test, y_train, y_test, model )

# 4. Show model generalization
miRNA = FoldCrossValidation(scaled_df, model)

# 5. Generate model with Feature selection
X_train = X_train.loc[:,miRNA]
X_test = X_test.loc[:,miRNA]
modelSelection = ModelFitPrediction (X_train, X_test, y_train, y_test, model )


# 6. 3 times Upsampling on AIS and Normal group

up_df = pd.DataFrame()
for c in ['AIS', 'Normal']:
    sampling = data[data['Class'] == c]
    up_sampling = resample(sampling, replace=True, n_samples=sampling.shape[0]*3, random_state=123)
    up_df = up_df.append(up_sampling)

hsil  = data[data['Class']== 'HSIL']

new_data = hsil.append(up_df)

# 7. Show data description
DataDescription(data=new_data)
scaled_df = Normalization(data=new_data)
# 8. Train and Test model 
# Train and Test spliting with 0.67 and 0.33 respectively
X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:,:-1], scaled_df['Target'], 
                                                    test_size=0.33) # , random_state=7
model=RandomForestClassifier() # step 1: choose model/estimator for classification
model = ModelFitPrediction (X_train, X_test, y_train, y_test, model )

# 9. Show model generalization
miRNA = FoldCrossValidation(scaled_df, model)

# 10. Generate model with Feature selection
X_train = X_train.loc[:,miRNA]
X_test = X_test.loc[:,miRNA]
modelSelection = ModelFitPrediction (X_train, X_test, y_train, y_test, model )

output = base + 'Importance_Feature.csv'
miRNA.to_csv(output)





