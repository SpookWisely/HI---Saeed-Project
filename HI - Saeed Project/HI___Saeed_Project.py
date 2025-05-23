import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
##CSV importing.
weather_datal7 = pd.read_csv(r'C:\Users\\Harry\source\\repos\HI - Saeed Project\HI - Saeed Project\L7Weather.csv') ##Have to Change the File path for other pcs
weather_datal9 = pd.read_csv(r'C:\Users\\Harry\source\\repos\HI - Saeed Project\HI - Saeed Project\L9Weather.csv')

##print(weather_dataL7.head(6) ,"\n")
##print(weather_dataL9.head(14))
##Dropping the day column as I feel it wasn't needed for the training.
weather_datal9 = weather_datal9.drop(columns =['Day'])
##Variable Declaration for the split of the data to be used in the models split.
X = weather_datal9.iloc[:,0:5] 
print(X,'\n')
y = weather_datal9.iloc[:,-1: ]
print(y,'\n')
 
##oneHotEncoder for natural language to be used in the models.
oneHot = OneHotEncoder(sparse_output = False).set_output(transform='pandas')

X_encoded = oneHot.fit_transform(X[['Outlook','Temp','Humidity','Wind']])
print(X_encoded,'\n')
y_Encoded = oneHot.fit_transform(y)
##Similar to one hot encoder but keeps the results within the same column.
labelEncoder = LabelEncoder()
y_Encoded = labelEncoder.fit_transform(y['Tennis'])  # Encode "Yes" as 1 and "No" as 0
y_HEncoded = oneHot.fit_transform(y[['Tennis']])
print(y_Encoded,'\n')


def decisonTreeFunction(X,y,X_encoded,y_Encoded):
##Splitting the data into training and testing data.
    X_train, X_test, y_Train, y_Test = train_test_split(X_encoded, y,random_state=7, test_size=0.33)
##Printing Out The shape of data to see what's what.
    print(X_train.shape,'\n')
    print(X_test.shape,'\n')
    print(y_Train.shape,'\n')
    print(y_Test.shape,'\n')
##Decison Tree Declaration.
    decTree = DecisionTreeClassifier()
    decTree.set_params(criterion='entropy',ccp_alpha =0.02, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=7)
    decTree.fit(X_train, y_Train)
##Prediction declaration
    y_pred = decTree.predict(X_test)
## Saw Online good to make use of a confusion matrix not too sure on the results given.
    print(confusion_matrix(y_Test,y_pred),'\n')
##Classifcation report declaration for the model to see importance of values.
    print(classification_report(y_Test, y_pred),'\n')
    X.columns
    decTree.feature_importances_ 
##Feature importance to see which features are most important in the model.
    features = pd.DataFrame(decTree .feature_importances_, index = X_encoded.columns, columns=['Importance'])
    print(features.sort_values(by='Importance', ascending=False),'\n')
    
    return

def extraTreeFunction(X,y,X_encoded,y_HEncoded):

    X_train, X_test, y_Train, y_Test = train_test_split(X_encoded, y_HEncoded,random_state=7, test_size=0.33)
    
    extraTree = ExtraTreesClassifier(random_state=21) ##default extratree
 ##fit method which is used to fit data to the model to be used in the tree.
    extraTree.fit(X_train, y_Train)

 ##cross_val_score(extraTree, X_train, y_Train, cv=5, scoring = 'accuracy', n_jobs=-1).mean()
    print('ExtraTree Result with Default Settings: ',cross_val_score(extraTree, X_train, y_Train, cv=5, scoring = 'accuracy', n_jobs=-1,).mean(),'\n')

 ##y_pred = extraTree.predict(X_test)
 ##Params to be used for the non default extratree
    param_grid = {
        'criterion': ['entropy'],
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2,],
        'min_samples_leaf': [1, 2],}
    extraTreeUpdated = GridSearchCV(extraTree, param_grid, cv=5, n_jobs=-1)
    extraTreeUpdated.fit(X_train, y_Train)
    extraTreeUpdated.best_params_
    print('ExtraTree Result With NonDefault Settings Using Params:',extraTreeUpdated.best_score_,'\n')

    return



def randomForestFunction(X,y,X_encoded,y_Encoded):
    
    X_train, X_test, y_Train, y_Test = train_test_split(X_encoded, y_Encoded,random_state=7, test_size=0.33)
 ##Default Random Forest Declaration.
    randFor = RandomForestClassifier()

    randFor.fit(X_train, y_Train)

    y_Pred= randFor.predict(X_test)
   
    print('Score for default Random Forest: ',randFor.score(X_test, y_Test), '\n')
    print(classification_report(y_Test, y_Pred),'\n')
    features = pd.DataFrame(randFor.feature_importances_, index = X_encoded.columns, columns=['Importance'])
    features.sort_values(by='Importance', ascending=False)
    print(features,'\n')
 ##Random Forest with non default settings but with entropy criterion chosen.
    randForTwo = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=11, min_samples_split=3, min_samples_leaf=1, random_state=7)
    
    randForTwo.fit(X_train, y_Train)

    y_Pred= randForTwo.predict(X_test)
    print('Score for entropy Random Forest: ',randForTwo.score(X_test, y_Test), '\n')
    
    
    featuresNonDe = pd.DataFrame(randForTwo.feature_importances_, index = X_encoded.columns, columns=['Importance'])
    featuresNonDe.sort_values(by='Importance', ascending=False)
    print(featuresNonDe,'\n')
    return


##Uncomment these to test each method for each data model.

##extraTreeFunction(X,y,X_encoded,y_HEncoded)
##decisonTreeFunction(X,y,X_encoded,y_Encoded)
##randomForestFunction(X,y,X_encoded,y_Encoded)