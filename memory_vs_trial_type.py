#Data Processing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

#function to find the trial tyoes for the current trial given the pid and picid
def find_curr_trial (mem_pid, mem_picid):
    for i in range (1, len(control)):
        if control[i][0] == mem_pid:
            if control[i][2] == mem_picid:
                return (control[i][5], control[i][6])
    
    return('-','-') #if this fails for any reason


#determines ratio of "1" responses to total responses in that category
def accuracy(response_list):
    accurate = [i for i in response_list if i == 1]
    return(len(accurate)/len(response_list))


#find accuracy for the individual subject
def subject_accuracy(curr_ID):
    #initialize arrays for different trial types
    congruent_go = []
    congruent_nogo = []
    incongruent_go = []
    incongruent_nogo = []
    
    #find memory accuracy in relation to previous trial type
    for i in range (0, len(memory)):
        if (memory[i][0] == curr_ID) and (memory[i][5] == "old"):
            prev_conditions = find_curr_trial(memory[i][0], memory[i][2])
            response = memory[i][10]
            if prev_conditions[0] != '-':
                if prev_conditions[0] == "congruent":
                    if prev_conditions[1] == "go":
                        congruent_go.append(response)
                    else:
                        congruent_nogo.append(response)
                else:
                    if prev_conditions[1] == "go":
                        incongruent_go.append(response)
                    else:
                        incongruent_nogo.append(response)
    
    accuracies = [accuracy(congruent_go), accuracy(congruent_nogo), accuracy(incongruent_go), accuracy(incongruent_nogo)]
    return accuracies

#Responses will be contained in this two dimensional list so that it is in the 
# order of congruent_go, congruent_nogo, incongruent_go, incongruent_nogo lists
all_accuracies = [[],[],[],[]]
ANOVA_list = []

#get accuracies for each participant and then put into all_accuracies 
for ID in pid_list:
    individ_acc = subject_accuracy(ID)
    ANOVA_list.append([ID, 'go', 'congruent', individ_acc[0]])
    ANOVA_list.append([ID, 'nogo', 'congruent', individ_acc[1]])
    ANOVA_list.append([ID, 'go', 'incongruent', individ_acc[2]])
    ANOVA_list.append([ID, 'nogo', 'incongruent', individ_acc[3]])
    
    for i in range(4):
        all_accuracies[i].append(individ_acc[i])

#ANOVA
data = pd.DataFrame(ANOVA_list, columns = ['pid', 'response', 'congruency', 'SbjACC'])

gpResult = data.groupby(['response','congruency']).SbjACC.mean().reset_index()
print(gpResult)

curr_ANOVA = AnovaRM(data, 'SbjACC', 'pid', within = ['response', 'congruency'])
curr_ANOVA = curr_ANOVA.fit()
print(curr_ANOVA)
        
#Overall Analysis
all_accuracies_average = []

for i in range(len(all_accuracies)):
    all_accuracies_average.append(sum(all_accuracies[i])/len(all_accuracies[i]))
    
print(all_accuracies_average)

all_accuracies_average = [all_accuracies_average[0], all_accuracies_average[2], all_accuracies_average[1], all_accuracies_average[3]]
#plot the relationship

labels = (' congruent\ngo ',  ' incongruent\ngo ', ' congruent\nnogo ', ' incongruent\nnogo ')
y_pos = np.arange(len(labels))

fig = plt.figure(1)
'''
plt.bar(y_pos, all_accuracies_average)
plt.xticks(y_pos, labels)
plt.ylabel("Memory Accuracy")
plt.xlabel("Trial Type")
plt.title("Memory Accuracy vs. Trial Type")
for i in range(len(all_accuracies_average)):
    plt.text(x=y_pos[i] - 0.12, y=all_accuracies_average[i] + 0.01, \
             s=round(all_accuracies_average[i], 3), size=10)
plt.show()

plt.figure(2)
'''
plt.bar(y_pos, all_accuracies_average, color="gold")
plt.xticks(y_pos, labels)
plt.ylabel("Memory Accuracy")
plt.xlabel("Trial Type")
plt.title("Memory Accuracy vs. Trial Type")
plt.ylim(.4, .7)
for i in range(len(all_accuracies_average)):
    plt.text(x=y_pos[i] - 0.12, y=all_accuracies_average[i] + 0.01, \
             s=round(all_accuracies_average[i], 3), size=10)
plt.show()
fig.savefig("memory_vs_trial_type.png")


