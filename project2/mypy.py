# function MinimaxDecision(state) returns an action (pg 166)
def MinimaxDecision(state, Utility, maxDepth):

    pacmanIndex = 0
    firstGhostIndex = 1

    legalActions = state.getLegalActions(pacmanIndex)
    legalActions.remove('Stop')

    if(len(legalActions) == 0):
        print "No Legal Moves - Not sure if this is even possible"
        return None

    maxValuedAction = legalActions[0]
    maxValue = -10000000 # Assume all values will not be below this.

    for action in legalActions :
        tempValue = MinValue(state.generateSuccessor(pacmanIndex, action), firstGhostIndex, Utility, maxDepth)
        if(tempValue > maxValue):
            maxValue = tempValue
            maxValuedAction = action

    #print(maxValue)
    return maxValuedAction

# function MaxValue(state) returns a utility value (pg 166)
def MaxValue(state, Utility, depth):
    # Always decrement depth during Pacman's turn
    depth -= 1

    # Check if time to evaluate
    if TerminalTest(state,depth):
        return Utility(state)

    pacmanIndex = 0
    firstGhostIndex = 1

    # Remove stop from Pacman's list of possible moves for better speed at higher depth
    legalActions = state.getLegalActions(pacmanIndex)
    legalActions.remove('Stop')

    return max(MinValue(state.generateSuccessor(pacmanIndex, action), firstGhostIndex, Utility, depth) for action in legalActions)

# function MinValue(state) returns a utility value (pg 166)
def MinValue(state, ghostIndex, Utility, depth):
    # Check if time to evaluate
    if TerminalTest(state,depth):
        return Utility(state)

    numAgents = state.getNumAgents()
    nextGhostIndex = ghostIndex + 1
    legalActions = state.getLegalActions(ghostIndex)

    # Check for another ghost min layer
    if nextGhostIndex < numAgents:
        return min([MinValue(state.generateSuccessor(ghostIndex, action),nextGhostIndex, Utility, depth) for action in legalActions])
    # Else pacman max layer
    else:
        return min([MaxValue(state.generateSuccessor(ghostIndex, action), Utility, depth) for action in legalActions])

# Terminal Test for use with MinValue and MaxValue
def TerminalTest(state,depth):
    if depth == 0 or state.isWin() or state.isLose():
        return True
    else:
        return False

# CSE 511A Lecture 6 -- Adversarial Search.pptx (Slide 39)
def AlphaBetaMiniMaxDecsion(state, Utility, maxDepth):
    pacmanIndex = 0
    firstGhostIndex = 1

    legalActions = state.getLegalActions(pacmanIndex)
    legalActions.remove('Stop')

    if(len(legalActions) == 0):
        print "No Legal Moves - Not sure if this is even possible"
        return None

    maxValuedAction = legalActions[0]
    alpha = -10000000 # Synonymous with max value for this first max layer
    beta = 10000000 # Ignored in this first layer

    for action in legalActions :
        tempValue = AlphaBetaMinValue(state.generateSuccessor(pacmanIndex, action), firstGhostIndex, Utility, maxDepth, alpha, beta)
        if(tempValue > alpha):
            alpha = tempValue
            maxValuedAction = action

    print(alpha)
    return maxValuedAction

# CSE 511A Lecture 6 -- Adversarial Search.pptx (Slide 39)
def AlphaBetaMaxValue(state, Utility, depth, alpha, beta):
    # Always decrement depth during Pacman's turn
    depth -= 1

    # Check if time to evaluate
    if TerminalTest(state,depth):
        return Utility(state)

    pacmanIndex = 0
    firstGhostIndex = 1

    # Remove stop from Pacman's list of possible moves for better speed at higher depth
    legalActions = state.getLegalActions(pacmanIndex)
    legalActions.remove('Stop')

    v = -1000000

    for action in legalActions:
        v = max(v, AlphaBetaMinValue(state.generateSuccessor(pacmanIndex, action),firstGhostIndex, Utility, depth, alpha, beta))
        if(v >= beta):
            return v
        alpha = max(alpha,v)

    return v

# CSE 511A Lecture 6 -- Adversarial Search.pptx (Slide 39)
def AlphaBetaMinValue(state, ghostIndex, Utility, depth, alpha, beta):
    # Check if time to evaluate
    if TerminalTest(state,depth):
        return Utility(state)

    numAgents = state.getNumAgents()
    nextGhostIndex = ghostIndex + 1
    legalActions = state.getLegalActions(ghostIndex)

    v = 10000000

    # Check for another ghost min layer
    if nextGhostIndex < numAgents:
        for action in legalActions:
            v = min(v, AlphaBetaMinValue(state.generateSuccessor(ghostIndex, action), nextGhostIndex, Utility, depth, alpha, beta))
            if(v <= alpha):
                return v
            beta = max(beta,v)

        return v
    # Else pacman max layer
    else:
        for action in legalActions:
            successor = state.generateSuccessor(ghostIndex, action)
            v = min(v, AlphaBetaMaxValue(state.generateSuccessor(ghostIndex, action), Utility, depth, alpha, beta))
            if(v <= alpha):
                return v
            beta = max(beta,v)

        return v
