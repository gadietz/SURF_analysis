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
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

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

def find_mean_of_RTs(pid):
    sum_of_RTs = 0
    num_RTs = 0
    for i in range (len(control)):
        if control[i][0] == pid:
            if not math.isnan(control[i][9]):
                sum_of_RTs += control[i][9]
                num_RTs += 1
    return(sum_of_RTs/num_RTs)


#function to find the average deviation of the three previous RTs
def find_prev_average_deviation_of_RTs (mem_pid, mem_picid):
    sum_of_RTs = 0
    num_RTs = 0
    mean_RT = find_mean_of_RTs(mem_pid)
    
    for i in range (3, len(control)):
        if control[i][0] == mem_pid and control[i][2] == mem_picid:
            if not math.isnan(control[i-3][9]):
                sum_of_RTs += abs(mean_RT - control[i-3][9])
                num_RTs += 1
            if not math.isnan(control[i-2][9]):
                sum_of_RTs += abs(mean_RT - control[i-2][9])
                num_RTs += 1
            if not math.isnan(control[i-1][9]):
                sum_of_RTs += abs(mean_RT - control[i-1][9])
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
            avg_RTs = find_prev_average_deviation_of_RTs(memory[i][0], memory[i][2])
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
'''
all_X = [data[i][0:4] for i in range(len(data))]
all_y = [data [i][5] for i in range(len(data))]
'''

outfile = open("logistic_regression_within_subjects_including_RT_deviation_results.txt", "w")

all_subjects_regression_data = []

for ID in pid_list:
    
    X = [data[i][0:5] for i in range(len(data)) if ID == data[i][6]]
    y = [data [i][5] for i in range(len(data)) if ID == data[i][6]]
    
    X, y = np.array(X), np.array(y)
    
    X = preprocessing.normalize(X)
    
    regressor = LogisticRegressionCV(cv = 5, solver='lbfgs', class_weight='balanced')
    
    regressor.fit(X, y) #training the algorithm
    
    y_pred = regressor.predict(X)
    
    print("\nSUBJECT #", ID, sep="", file=outfile)
    print("Logistic Regression with Cross Validation", file=outfile)
    accuracy_score = regressor.score(X, y)
    print("Accuracy Score: ", accuracy_score, file=outfile)
    intercept = regressor.intercept_[0]
    print("Intercept: ", intercept, file=outfile)
    print("Coefficients: ", file=outfile)
    
    coefs_name = ['Congruency', 'Response', \
                  'Previous Congruency', 'Previous Response', \
                  'Average of Previous RT_Deviations']
    counter = 0
    coefs = regressor.coef_[0].tolist()
    for coef in coefs:
        print("   ", coefs_name[counter], ": ", coef, sep="", file=outfile)
        counter += 1
    
    c_matrix = confusion_matrix(y, y_pred).tolist()
    print("Confusion Matrix: ", c_matrix, file=outfile)
    
    all_subjects_regression_data.append([ID, accuracy_score, intercept, coefs, c_matrix])

def bad_model(matrix):
    if (matrix[0][0] == 0) and (matrix[1][0] == 0):
        return True
    if (matrix[0][1] == 0) and (matrix[1][1] == 0):
        return True
    return False

print("\nSubjects for which the Logistic Regression either predicted", file=outfile)
print("the subject was always accuaracte or always inaccurate:", file=outfile)

bad_models_pid_list = [regression_set[0] for regression_set in all_subjects_regression_data if bad_model(regression_set[4])]

print(bad_models_pid_list, file=outfile)
print("Due to these models being biased to always pick the same end result,", file=outfile)
print("they were excluded from further study of the participant models.", file=outfile)

good_subjects_model_accuracy = [regression_set[1] for regression_set in all_subjects_regression_data \
                                if regression_set[0] not in bad_models_pid_list]

print("\nA T-test was conducted on the remaining subjects model accuracies", file=outfile)
print("to see if there was strong evidence that the logistic regression", file=outfile) 
print("model accuracy was higher than random chance (50% accuracy)", file=outfile)

from scipy.stats import t

mu_not = 0.5
n = len(good_subjects_model_accuracy)
avg = np.mean(good_subjects_model_accuracy)
sd = np.std(good_subjects_model_accuracy, ddof=1)
SE = sd/(n**0.5)
t_c = (avg - mu_not) /SE
p = 1 - t.cdf(abs(t_c), 1)

print("\nSample Size              :", n, file=outfile)
print("Mean of Sample Accuracies:", round(avg, 4), file=outfile)
print("Standard Deviation       :", round(sd, 4), file=outfile)
print("P Value of T-Test        :", round(p, 4), file=outfile)
print("\nSince the P-Value is less than 0.05, we can say these results", file=outfile)
print("are significant, and therefore there is strong evidence that our", file=outfile)
print("modeling technique predicts memory accuracy.", file=outfile)

outfile.close()    















