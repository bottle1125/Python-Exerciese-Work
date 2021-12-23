import matplotlib.pyplot as plt
import math
import numpy
numpy.set_printoptions(precision =2)
import scipy.stats as stats
import statsmodels.api as sm
#### During the course, we derived a model where, for a target of size W that is D away, the time required to select it is MT = b log_2 (1 + D/W). We are going to use this to model keyboard typing with a single finger.







########################### Modeling keyboard



## Define our keyboard

line1 = 'qwertyuiop'
line2 = 'asdfghjkl'
line3 = 'zxcvbnm'


### Define a keyboard as a list of keys. Each key also mentions its position in the keyboard (line number, column number)
keyboard  = [(i, (0,ni)) for ni,i in enumerate(line1)] + [(i, (1,ni)) for ni,i in enumerate(line2)] + [(i, (2,ni)) for ni,i in enumerate(line3)]


### Define empty matrices to hold the results
ids = numpy.zeros((26,26))
Ds = numpy.zeros((26,26))
W = 1


alphabet = line1 + line2 + line3

### Get Ids for each key combination
for ns,keystart in enumerate(alphabet):
    for ne, keyend in enumerate(alphabet):
        xs,ys, xe,ye = keyboard[ns][1][1] + 2/5*keyboard[ns][1][0], keyboard[ns][1][0], keyboard[ne][1][1] + 2/5*keyboard[ne][1][0], keyboard[ne][1][0]
        ## Compute euclidian distance between the startkey and endkey
        D = math.sqrt((xe - xs)**2 + (ye - ys)**2)
        Ds[ns,ne] = D
        ## Use Fitts' law formula
        ids[ns,ne] = math.log(1 + D / W)





### This function can be used to print out an array in a form suited for a LaTex table. You can adapt it to print out the array in a different format if needed.

def bmatrix(a):
    text = r'$\left[\begin{array}{*{'
    text += str(len(a[0]))
    text += r'}c}'
    text += '\n'
    for x in range(len(a)):
        for y in range(len(a[x])):
            text += '{:.2f}'.format(a[x][y])
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    text += r'\end{array}\right]$'
    print(text)


print(ids)

################################## Estimating Model parameters



## This function opens the file filename, and reads the startkey, endkey, and duration of each typing stroke


##### load the array of ID using numpy.load() if you didn't succeed in creating it
# ids = numpy.load(#TODO)

#### Use this function to open the file keystrokes.csv. It returns the list of ID and MT's
### Inputs: filename: name of the file you want to open
###         ids: array of IDs for the keyboard
def get_keystrokes(filename, ids):
    ID, MT = [], []
    with open(filename, 'r') as _file:
        startime = 0
        for n,line in enumerate(_file):
            print(n,line)
            try:
                startkey, stopkey, time = line.split(",")
                print(startkey, stopkey, time)
                time = float(time.rsplit('\n')[0])
                # if startkey == 'None':
                if n == 0:
                    startime = time
                    continue
                startkey = startkey.split("'")[1]
                stopkey = stopkey.split("'")[1]
                if startkey not in alphabet or stopkey not in alphabet:
                    continue
                ns = alphabet.index(startkey)
                ne = alphabet.index(stopkey)
                mt = time - startime
                MT.append(mt)
                ID.append(ids[ns,ne])
                startime = time
            except IndexError:
                pass
    return ID,MT



### Complete this function to estimate the parameters of Fitts' law
def analyse_keystrokes(ID,MT,ax):
    ax.plot(ID,MT,'o')


    #TODO: complete this function to estimate the slope and intercept parameters. Report Goodness of fits for Fitts' model, as well as a report on the confidence of the estimated parameters. You can use the scipy.stats or statsmodels libraries, or any other tool you prefer.
    return


#### Show how well Fitts' model fits: graphical representation, a goodness of fit of choice, and the uncertainty with regards to estimated parameters

fig = plt.figure()
ax = fig.add_subplot(111)
###TODO: use the get_keystrokes fucntion to get the IDs associated MT for each keystroke.
ID,MT = get_keystrokes(##TODO)
####TODO: finish and use the analyse_keystrokes function to fit and plot Fitts' model.
##TODO= analyse_keystrokes(ID,MT,ax)

plt.show()



################################# Most Common words

## Estimate the average time needed to type the 1000 most common english words. These words are given in the accompanying mostcommonwords.txt file. No information of frequency is given, so assume that each word is equally probable. Consider that the first key is "free", i.e. time starts only after the first key of the word is pressed.

slope = 0.12
intercept = 0.03

with open("most_common_words.txt", 'r') as _file:
    T = []
    for nline, line in enumerate(_file):
        t = 0 ### initialize the typing time for a word
        line = line[:-1]
        print(line, len(line))
        for i in range(0, len(line)-1):
            try:
                ns = alphabet.index(line[i])
                ne = alphabet.index(line[i+1])
                id = ids[ns,ne]
                ###TODO: increment typing time for each letter stroke
                t +=
            except ValueError:
                pass
        T.append(t)

print("Average time needed")
#TODO: compute the average typing time from the list of typing times T. You can use e.g. numpy's mean function
print(#TODO)

fig = plt.figure()
ax = fig.add_subplot(121)
axhist = fig.add_subplot(122)

ax.plot(range(0,len(T)), T, 'o')
ax.set_ylabel("Typing time (seconds)")
ax.set_xlabel("Word Index")
###TODO: Plot the histogram of the typing times (use plt.hist() function )
axhist.hist(#TODO)
axhist.set_xlabel(#TODO)
axhist.set_ylabel(#TODO)
plt.show()
plt.close()


############################### Modeling word frequency

## Here are some frenquency / occurence pairs for the 100 most frequent english words. Use it to estimate s.

frequency = [1e9, 182e6, 99e6, 73e6, 53e6, 41e6, 34e6, 30e6, 25.7e6, 22.5e6, 20e6]
rank  = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


logfreq = [math.log(f,2) for f in frequency]
logoc = [math.log(o,2) for o in rank]

###TODO Fit Zipf's model to the frequencey-rank relationship, as explained in the assigment.

## Plot the data + fitted model
fig = plt.figure()
ax = fig.add_subplot(121)
axlog = fig.add_subplot(122)
ax.plot(rank, frequency, '-')

axlog.plot(#TODO)
plt.show()


## Use this new information to evaluate the average time it takes to type the 1000 most common words
## Assume that there are only 1000 words in total, i.e. N = 1000. Skip the case of single letter words
s = 0.85

##TODO: Compute the normalizing sum (denominator) in the p(k) expression, with s = 0.85
_sum = #TODO

## TODO: compute the p(k)'s for each k
weights = []
for nt in range(1,1001):
    weight = #TODO/_sum
    weights.append(weight)

weighted_sum = 0
for w,t in zip(weights, T):
    weighted_sum += w*t

print(weighted_sum)


######################################## New keyboard layout

#TODO: Modify the definition of the keyboard given above. Then you should be able to re-use some of the functions we used before.
