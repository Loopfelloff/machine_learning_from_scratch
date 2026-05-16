# This is md is gonna contain my pseudo code for the dual method using SMO.
```
main routine():
    numChanged = 0 # we try to find how many did we change.
    examineAll = True # should we examine all (boolean value)
    while (numChanged > 0 or examineAll):
        numChanged  = 0
        if(examineAll):
            loop I over all training examples
            numChanged += examineExample(I) # this returns either one or 0
        else : 
            loop I over examples where alpha is not 0 and not C meaning for bound ones, meanig if everything doesn't have to be examined we select a portion to examine
        if examineAll = True:
            examineAll = false
        elif numChanged ==0 : meaning even after examinig All nothing has changed set examineAll to 1

```

This while loop only breaks after when examinigAll still no alpha is changed

meaning after we tried to first iterate over the value of alpha1 and change some, 
after those change the non bound is also changed
after that it is seen that no change is required further more after examineAll is set to true 
and numChanged still remains 0 meaning no furhter change was made all the terms fall within 1-error.

## examineExample sub-routine

```

sub_routine examineExample(i2):
    y2 = target[i2] # meaning the label ofcourse.
    alph2 = Lagrange multiplier for i2
    E2 = SVM output on point[i2]  - y2 (use cache)
    r2 = E2 * y2
    if ((r2 <-tol and alph2 < C ) or (r2 > tol and alph2 > 0))
    {
        i1 = result of second choice heuristic # here only the maximixing E is talked about.
        if takeStep(i1 ,i2)
            return 1
}
    loop over all non-zero and non-C alhpa , starting at a random point{
        i1= identity of current alpha
        if takeStep(i1, i2):
            return 1
}

    loop over all possible i1 , starting at a random point {
        i1 = loop variable
        if takeStep(i1 , i2)
            return 1
}
    loop over all possible i1 , starting at a random point {
        i1 = loop variable  
        if (takeStep(i1, i2))
            return 1
}
    return 0

```
The above one felt pretty straight forward.

## takeStep sub-routine

```
sub_routine takeStep(i1, i2):
if(i1 == i2) return 0 # of course
alph1 = Lagrange multipler for i1
y1 = target[i1]
E1 = SVM output on point[i1] - y1
s = y1 * y2
Computer L , H with the max and min equation
if(L ==H) return 0
k11 
k22
k12
eta 
if(eta > 0)
{
a2 = the equation for a2
clipt a2
}
else{
Lob j = objective funct at a2 = L
Hobj = objective function at a2 = H
if (Lobj < Hobj-eps) # eps is a small valeu like 0.001
    a2 =L
else if (Lobj > Hobj + eps)
    a2 - H
else 
    a2 =alph2
}
if (|a2 - alhp2| < eps * (a2 + alph2 + eps)) # this is to ensure positive change
    return 0 
a1 = alph1+s*(alph2-a2)
Update threshold to reflect change in Lagrange multipliers
Update weight vector to reflect change in a1 & a2, if SVM is linear
Update error cache using new Lagrange multipliers
Store a1 in the alpha array
Store a2 in the alpha array
return 1
}
```



        

