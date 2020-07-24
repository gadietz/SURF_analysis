
#Data Processing
import pandas as pd
import matplotlib.pyplot as plt
import math
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
    #initialize array for RTs and accuracies
    RT_and_accuracy = []
    
    #find memory accuracy in relation to previous RTs
    for i in range (1, len(memory)):
        if (memory[i][0] == curr_ID) and (memory[i][5] == "old"):
            avg_RTs = find_prev_average_deviation_of_RTs(memory[i][0], memory[i][2])
            accuracy = memory[i][10]
            if avg_RTs != '-':
                RT_and_accuracy.append([avg_RTs, accuracy, 1-accuracy, memory[i][0], memory[i][2]])
        
    return RT_and_accuracy

#Responses will be contained here
accuracies_and_avg_RTs = []

#get accuracies for each participant and then put into all_accuracies 
for ID in pid_list:
    accuracies_and_avg_RTs += subject_accuracy(ID)
    
accuracies_and_avg_RTs_DataFrame = \
pd.DataFrame(accuracies_and_avg_RTs, columns = ['avg_RT_Deviation', 'sbjACC', 'ER', 'pid', 'picid'])


#plot the relationship between Average RTs and memory accuracy
plt.plot(accuracies_and_avg_RTs_DataFrame['avg_RT_Deviation'], \
         accuracies_and_avg_RTs_DataFrame['sbjACC'], 'o')
plt.ylabel("Accuracy")
plt.xlabel("Average RT Deviation")
plt.title("Memory Accuracy vs. Average RT Deviation")
plt.show()


def avg_list(lst):
    return sum(lst)/len(lst)


ANOVA_list = []

for ID in pid_list:
    
    prev_RTs_correct = []
    prev_RTs_incorrect = []
    
    for data in accuracies_and_avg_RTs:
        if data[3] == ID:
            if data[1] == 1:
                prev_RTs_correct.append(data[0])
            else:
                prev_RTs_incorrect.append(data[0])
                

    ANOVA_list.append([ID, 1,  avg_list(prev_RTs_correct)])
    ANOVA_list.append([ID, 0,  avg_list(prev_RTs_incorrect)])

#ANOVA
ANOVA_DataFrame = pd.DataFrame(ANOVA_list, columns = ['pid', 'sbjACC', 'avg_RT_Deviation'])

gpResult = ANOVA_DataFrame.groupby(['sbjACC']).avg_RT_Deviation.mean().reset_index()
print()
print(gpResult)
print()
ANOVA = AnovaRM(ANOVA_DataFrame, 'avg_RT_Deviation', 'pid', within = ['sbjACC'])
ANOVA = ANOVA.fit()
print(ANOVA) 



















