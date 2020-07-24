#Data Processing
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import normalize
import sklearn.naive_bayes as nb

memory = pd.read_csv('data_memory.csv')
control = pd.read_csv('data_control.csv')

#remove bad subjects - found previously
excludedSubjects = [5, 21, 38, 72] #pids of bad subjects

#remove bad subjects from memory dataframe
bad_mem_indices = []
for indx in range (0, len(memory['pid'])):
    curr_pid = memory['pid'][indx]
    is_bad = False
    for subject in excludedSubjects:
        if curr_pid == subject:
            is_bad = True
    if is_bad:
        bad_mem_indices.append(indx)
memory = memory.drop(bad_mem_indices)

#remove bad subjects from control dataframe
bad_con_indices = []
for indx in range (0, len(control['pid'])):
    curr_pid = control['pid'][indx]
    is_bad = False
    for subject in excludedSubjects:
        if curr_pid == subject:
            is_bad = True
    if is_bad:
        bad_con_indices.append(indx)
control = control.drop(bad_con_indices)

pid_list = memory.pid.unique().tolist()

memory = memory.values.tolist()
control = control.values.tolist()

#determines ratio of "1" responses to total responses in that category
def accuracy(response_list):
    accurate = [i for i in response_list if i == 1]
    return(len(accurate)/len(response_list))
    
def binarize_response(response):
    if response == "congruent":
        return(1)
    if response == "incongruent":
        return(0)
    if response == "go":
        return(1)
    if response == "nogo":
        return(0)  

#function to find the trial tyoes for the current trial given the pid and picid
def find_curr_trial (mem_pid, mem_picid):
    for i in range (1, len(control)):
        if control[i][0] == mem_pid:
            if control[i][2] == mem_picid:
                return (binarize_response(control[i][5]), \
                        binarize_response(control[i][6]))
    
    return('-','-') #if this fails for any reason

#function to find the trial tyoes for the previous trial given the pid and picid
def find_prev_trial (mem_pid, mem_picid):
    for i in range (1, len(control)):
        if control[i][0] == mem_pid and control[i][2] == mem_picid and i != 0:
                return (binarize_response(control[i-1][5]),\
                        binarize_response(control[i-1][6]))
    
    return('-','-') #if this fails for any reason

#function to find the average of three previous RTs
def find_average_prev_3_RTs (mem_pid, mem_picid):
    sum_of_RTs = 0
    num_RTs = 0
    
    for i in range (3, len(control)):
        if control[i][0] == mem_pid and control[i][2] == mem_picid:
            if not math.isnan(control[i-3][9]):
                sum_of_RTs += control[i-3][9]
                num_RTs += 1
            if not math.isnan(control[i-2][9]):
                sum_of_RTs += control[i-2][9]
                num_RTs += 1
            if not math.isnan(control[i-1][9]):
                sum_of_RTs += control[i-1][9]
                num_RTs += 1
    if num_RTs == 0:
        return('-') #basically if this fails for any reason
    else:
        return(sum_of_RTs/num_RTs)


#find accuracies paired with RTs for the individual subjects
def subject_accuracy(curr_ID):
    #initialize array for RTs, prev and curr trial types and accuracies
    data_collection = []
    
    #find memory accuracy in relation to previous RTs
    for i in range (1, len(memory)):
        if (memory[i][0] == curr_ID) and (memory[i][5] == "old"):
            curr_trial_type = find_curr_trial(memory[i][0], memory[i][2])
            prev_trial_type = find_prev_trial(memory[i][0], memory[i][2])
            avg_RTs = find_average_prev_3_RTs(memory[i][0], memory[i][2])
            accuracy = memory[i][10]
            
            if avg_RTs != '-' and curr_trial_type[0] != '-' and prev_trial_type[0] != '-':
                data_collection.append([curr_trial_type[0], curr_trial_type[1], \
                        prev_trial_type[0], prev_trial_type[1], \
                        avg_RTs, accuracy, memory[i][0], memory[i][2]])
    return data_collection

#Responses will be contained here
data = []

#get accuracies for each participant and then put into all_accuracies 
for ID in pid_list:
    data += subject_accuracy(ID)
 
'''       
data = pd.DataFrame(data, columns = ['congruency', 'response', \
                              'prev_congruency', 'prev_response', \
                              'avg_RTs', 'sbjACC', 'pid', 'picid'])
    

X = [[data['congruency'], data['response'],data['prev_congruency'], \
      data['prev_response'], data['avg_RTs']] for i in range(len(data['congruency']))]
y = [data ['sbjACC'] for i in range(len(data['congruency']))]
'''

X = [data[i][0:4] for i in range(len(data))]
y = [data [i][5] for i in range(len(data))]

#X = X.norma

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
regressor = nb.ComplementNB()
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

print("Naive Bayes")
print("Accuracy Score: ", regressor.score(X, y))
print("Test Score: ", regressor.score(X_test, y_test))
print("Test Accuracy Score: ", metrics.accuracy_score(y_test, y_pred))
print("Fl Score: ", metrics.f1_score(y_test, y_pred))
print("Parameters: ", regressor.get_params)
    
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix for Training Data:")
print(confusion_matrix(y_train, regressor.predict(X_train)))

print("Confusion Matrix for Testing Data")
print(confusion_matrix(y_test, y_pred))

'''
predictions_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
predictions_df1 = df.head(len(y_test))

df1.plot(kind='bar',figsize=(16,10))
plt.grid()
plt.show()
'''















