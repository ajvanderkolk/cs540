import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    X = {chr(x): 0 for x in range(ord('A'), ord('Z')+1)}
    
    with open (filename,encoding='utf-8') as f:
        for line in f:
            line = line.upper()
            for char in line:
                if char.isalpha():
                    try:
                        X[char] += 1
                    except KeyError:
                        #print(char)
                        continue

    return X

def hw2(letter_file, english_prior, spanish_prior):
    print("Q1")
    X = shred(letter_file)
    for key,val in X.items():
        print(key,val)
        
    print("Q2")
    param_vectors = get_parameter_vectors()
    e = param_vectors[0]
    s = param_vectors[1]
    X_log_e = X["A"] * math.log(e[0])
    X_log_s = X["A"] * math.log(s[0])
    print(f"{round(X_log_e,4):.4f}", f"{round(X_log_s,4):.4f}", sep="\n")

    print("Q3")
    F_e = math.log(english_prior) + sum([x * math.log(p) for x, p in zip(X.values(), e)])
    F_s = math.log(spanish_prior) + sum([x * math.log(p) for x, p in zip(X.values(), s)])
    print(f"{round(F_e,4):.4f}", f"{round(F_s,4):.4f}", sep="\n")

    print("Q4")
    #P_e = (math.e ** F_e) / ((math.e ** F_e) + (math.e ** F_s))
    #P_s = (math.e ** F_s) / ((math.e ** F_e) + (math.e ** F_s))
    Fe_Fs = F_s - F_e
    if Fe_Fs >= 100:
        P_e = 0.0
    elif Fe_Fs <= -100:
        P_e = 1.0
    else:
        P_e = 1 / (1 + (math.e ** Fe_Fs))
    print(f"{round(P_e,4):.4f}")



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

hw2(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))









