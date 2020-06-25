#Data Processing
import pandas as pd
memory = pd.read_csv('data_memory.csv')
control = pd.read_csv('data_control.csv')

#function to find the trial tyoes for the previous trial givne the pid and picid
def find_prev_trial (mem_pid, mem_picid):
    for i in range (1, len(control['pid'])):
        if control['pid'][i] == mem_pid:
            if control['picid'][i] == mem_picid:
                return (control['congruency'][i-1], control['response'][i-1])
    
    return('-','-') #if this fails for any reason

#initialize arrays for different trial types
congruent_go = []
congruent_nogo = []
incongruent_go = []
incongruent_nogo = []

#find memory accuracy in relation to previous trial type
for i in range (1, len(memory['pid'])):
    if memory['memCond'][i] == "old":
        print(i)
        prev_conditions = find_prev_trial(memory['pid'][i], memory['picid'][i])
        response = memory['sbjACC'][i]
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
  
#determines ratio of "1" responses to total responses in that category
def accuracy(response_list):
    accurate = [i for i in response_list if i == 1]
    return(len(accurate)/len(response_list))
    
#plot the relationship
import matplotlib.pyplot as plt
import numpy as np
labels = ('congruent go', 'congruent nogo', 'incongruent go', 'incongruent nogo')
y_pos = np.arange(len(labels))
accuracies = [accuracy(congruent_go), accuracy(congruent_nogo), accuracy(incongruent_go), accuracy(incongruent_nogo)]

plt.figure(1)
plt.bar(y_pos, accuracies)
plt.xticks(y_pos, labels)
plt.ylabel("Memory Accuracy")
plt.xlabel("N-1 Trial Type")
plt.title("Memory Accuracy vs. N-1 Trial Type")
for i in range(len(accuracies)):
    plt.text(x=y_pos[i], y=accuracies[i] + 0.01, s=round(accuracies[i], 3), size=10)
plt.show()

plt.figure(2)
plt.bar(y_pos, accuracies)
plt.xticks(y_pos, labels)
plt.ylabel("Memory Accuracy")
plt.xlabel("N-1 Trial Type")
plt.title("Memory Accuracy vs. N-1 Trial Type")
plt.ylim(.5, .6)
plt.show()

