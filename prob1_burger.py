
def dist(ingredientsOptimalDists, realDistances):
    d = 0
    for ingOptimalDist, dist in zip(ingredientsOptimalDists, realDistances):
        d += math.pow((ingOptimalDist-pos),2)
    return d

def getDistsToBun(k):
    if k%2 ==0:
        dists = range(0,k/2)
        dists.extend(xrange(k/2-1, -1, -1))
        return dists
    else:
        dists = range(0,(k-1)/2)
        dists.extend(xrange((k-1)/2, -1, -1))
        return dists

def minBurguerError(nIngredients, optimalDists):
    minError = -1
    # min error for considering ingredients up to index i
    minErrors = [-1]* nIngredients
    burger = [-1]* nIngredients  # needed to store?
    if len(optimalDists)=0:
        return 0
    elif len(optimalDists)==1:
        return optimalDists[0]**2
    else: # we can try put the ingredient at its optimal
        # distance, and since these are symetrical, there is 2 possible simmetric places
        # if one is taken in final burguer array.
        # fill burguer randomly only after optimal place is taken
        burger = [-1]* nIngredients  # needed to store?
        i = 0  # while there is ingredients left to add:
        while i< len(optimalDists):
            d = dist()
    return minErrors[nIngredients-1]


cases = int(raw_input())  # read a line with a single integer
#print 'cases: ',cases
for i in xrange(1, cases+1):
    nLevels = int(raw_input().split(" ")[0])
    listOfLevels = []
    #print 'nLevels: ',nLevels
    for level in range(nLevels): #      listOfLevels.append([n, experts]) #
        listOfLevels.append([int(s) for s in raw_input().split(" ")])  # read a list of integers, 2 in this case
    CEOLevel = minLevel(nLevels, listOfLevels)
    print "Case #{}: {} ".format(i, CEOLevel)
    # check out .format's specification for more formatting options
