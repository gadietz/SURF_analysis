#Data Processing
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import normalize
import sklearn.naive_bayes as nb
from statsmodels.stats.anova import AnovaRM

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

#function to find the trial tyoes for the current trial given the pid and picid
def find_curr_trial (mem_pid, mem_picid):
    for i in range (1, len(control)):
        if control[i][0] == mem_pid:
            if control[i][2] == mem_picid:
                return ((control[i][5]), \
                        (control[i][6]))
    
    return('-','-') #if this fails for any reason

#function to find the trial tyoes for the previous trial given the pid and picid
def find_prev_trial (mem_pid, mem_picid):
    for i in range (1, len(control)):
        if control[i][0] == mem_pid and control[i][2] == mem_picid and i != 0:
                return ((control[i-1][5]),\
                        (control[i-1][6]))
    
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
'''

def avg_list(lst):
    return sum(lst)/len(lst)
    
#find previous RTs for the individual subject
def subject_prev_RTs(curr_ID):
    #initialize arrays for different trial types
    congruent_go = []
    congruent_nogo = []
    incongruent_go = []
    incongruent_nogo = []
    
    #find memory accuracy in relation to previous trial type
    for trial in data:
        if curr_ID == trial[6]:
            avgRT = trial[4]
                
            if trial[0] == "congruent":
                if trial[1] == "go":
                    congruent_go.append(avgRT)
                else:
                    congruent_nogo.append(avgRT)
            else:
                if trial[1] == "go":
                    incongruent_go.append(avgRT)
                else:
                    incongruent_nogo.append(avgRT)
    
    previous_RTs = [avg_list(congruent_go), avg_list(congruent_nogo), \
                    avg_list(incongruent_go), avg_list(incongruent_nogo)]
    return previous_RTs


#conducting an ANOVAto compare prev_RT effect with response and congruency
ANOVA_list = []
all_RTs = [[], [], [], []]

#get accuracies for each participant and then put into all_accuracies 
for ID in pid_list:
    individ_RTs = subject_prev_RTs(ID)
    
    ANOVA_list.append([ID, 'go', 'congruent', individ_RTs[0]])
    ANOVA_list.append([ID, 'nogo', 'congruent', individ_RTs[1]])
    ANOVA_list.append([ID, 'go', 'incongruent', individ_RTs[2]])
    ANOVA_list.append([ID, 'nogo', 'incongruent', individ_RTs[3]])
    
    for i in range(4):
        all_RTs[i].append(individ_RTs[i])

#ANOVA
data = pd.DataFrame(ANOVA_list, columns = ['pid', 'response', 'congruency', 'Prev_RTs'])

gpResult = data.groupby(['response','congruency']).Prev_RTs.mean().reset_index()
print(gpResult)

ANOVA = AnovaRM(data, 'Prev_RTs', 'pid', within = ['response', 'congruency'])
ANOVA = ANOVA.fit()
print(ANOVA)    
        
#graphing averages
all_RTs_average = []

for i in range(len(all_RTs)):
    all_RTs_average.append(sum(all_RTs[i])/len(all_RTs[i]))
    
print(all_RTs_average)

#plot the relationship

labels = ('congruent go', 'congruent nogo', 'incongruent go', 'incongruent nogo')
y_pos = np.arange(len(labels))

plt.figure(1)
plt.bar(y_pos, all_RTs_average)
plt.xticks(y_pos, labels)
plt.ylabel("Previous Response Time")
plt.xlabel("Trial Type")
plt.title("Previous Response Time vs. Trial Type")
plt.ylim(0, 650)

for i in range(len(all_RTs_average)):
    plt.text(x=y_pos[i] - 0.25, y=all_RTs_average[i] + 20, \
             s=round(all_RTs_average[i], 3), size=10)
    
plt.show()






