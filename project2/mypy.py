import util
import os
import heapq
from game import Directions
from game import Agent
from game import Actions
from game import Grid
import time
import copy
import math


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

def ExpectedMiniMaxDecision(gameState, Utility, maxDepth):
  
  def maxLevel(gameState, depth):
    currDepth = depth + 1
    if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
      return Utility(gameState)
    maxvalue = -999999
    actions = gameState.getLegalActions(0)
    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      maxvalue = max(maxvalue, minLevel(successor, currDepth, 1))
    return maxvalue

  def minLevel(gameState, depth, agentIndex):
    minvalue = 999999
    if gameState.isWin() or gameState.isLose():
      return Utility(gameState)
    actions = gameState.getLegalActions(agentIndex)
    for action in actions:
      successor = gameState.generateSuccessor(agentIndex, action)
      if agentIndex == (gameState.getNumAgents() - 1):
        minvalue = min(minvalue, maxLevel(successor, depth))
      else:
        minvalue = min(minvalue, minLevel(successor, depth, agentIndex+1))
    return minvalue
    
    actions = gameState.getLegalActions(0)
    currentScore = -999999
    returnAction = ''
    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      score = minLevel(nextState, 0, 1)
      if score > currentScore:
        returnAction = action
        currentScore = score
    return returnAction

# function MaxValue(state) returns a utility value (pg 166)
def ExpectedMaxValue(state, Utility, depth):
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

  return max(ExpectedMinValue(state.generateSuccessor(pacmanIndex, action), firstGhostIndex, Utility, depth) for action in legalActions)

# function MinValue(state) returns a utility value (pg 166)
def ExpectedMinValue(state, ghostIndex, Utility, depth):
  # Check if time to evaluate
  if TerminalTest(state,depth):
    return Utility(state)

  numAgents = state.getNumAgents()
  nextGhostIndex = ghostIndex + 1
  legalActions = state.getLegalActions(ghostIndex)

  # Check for another ghost min layer
  if nextGhostIndex < numAgents:
    return sum([ExpectedMinValue(state.generateSuccessor(ghostIndex, action),nextGhostIndex, Utility, depth) for action in legalActions])/numAgents
  # Else pacman max layer
  else:
    return sum([ExpectedMaxValue(state.generateSuccessor(ghostIndex, action), Utility, depth) for action in legalActions])/numAgents

def mazeDistance(point1, point2, gameState):
  """
  Returns the maze distance between any two points, using the search functions
  you have already built.  The gameState can be any game state -- Pacman's position
  in that state is ignored.

  Example usage: mazeDistance( (2,4), (5,6), gameState)

  This might be a useful helper function for your ApproximateSearchAgent.
  """
  print point1, point2
  x1, y1 = point1
  x2, y2 = point2
  x1 = int(x1)
  x2 = int(x2)
  y1 = int(y1)
  y2 = int(y2)

  walls = gameState.getWalls()
  assert not walls[x1][y1], 'point1 is a wall: ' + point1
  assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
  prob = PositionSearchProblem(
      gameState, start=point1, goal=point2, warn=False)
  return len(astar(prob))

# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""


class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).

  You do not need to change anything in this class, ever.
  """

  def getStartState(self):
    """
    Returns the start state for the search problem
    """
    util.raiseNotDefined()

  def isGoalState(self, state):
    """
      state: Search state

    Returns True if and only if the state is a valid goal state
    """
    util.raiseNotDefined()

  def getSuccessors(self, state):
    """
      state: Search state

    For a given state, this should return a list of triples,
    (successor, action, stepCost), where 'successor' is a
    successor to the current state, 'action' is the action
    required to get there, and 'stepCost' is the incremental
    cost of expanding to that successor
    """
    util.raiseNotDefined()

  def getCostOfActions(self, actions):
    """
     actions: A list of actions to take

    This method returns the total cost of a particular sequence of actions.  The sequence must
    be composed of legal moves
    """
    util.raiseNotDefined()


def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
def generalSearch(problem, recorder):
  recorder.push([(problem.getStartState(), "Stop", 0)])
  visited = []
  while not recorder.isEmpty():
    path = recorder.pop()
    curr_state = path[-1][0]
    if problem.isGoalState(curr_state):
      return [x[1] for x in path][1:]
    if curr_state not in visited:
      visited.append(curr_state)
      for successor in problem.getSuccessors(curr_state):
        if successor[0] not in visited:
          successorPath = path[:]
          successorPath.append(successor)
          recorder.push(successorPath)
  return False


def depthFirstSearch(problem):
  return generalSearch(problem, util.Stack())


def breadthFirstSearch(problem):
  return generalSearch(problem, util.Queue())


def uniformCostSearch(problem):
  return generalSearch(problem, util.PriorityQueueWithFunction(
      lambda path: problem.getCostOfActions([x[1] for x in path][1:])))


def nullHeuristic(state, problem=None):
  return 0


def aStarSearch(problem, heuristic=nullHeuristic):
  return generalSearch(problem, util.PriorityQueueWithFunction(
      lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


class PositionSearchProblem(SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be
  used to find paths to a particular point on the pacman board.

  The state space consists of (x,y) positions in a pacman game.

  Note: this search problem is fully specified; you should NOT change it.
  """

  def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True):
    """
    Stores the start and goal.

    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    if start != None:
      self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print('Warning: this does not look like a regular search maze')

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    isGoal = state == self.goal

    # For display purposes only
    if isGoal:
      self._visitedlist.append(state)
      import __main__
      if '_display' in dir(__main__):
        # @UndefinedVariable
        if 'drawExpandedCells' in dir(__main__._display):
          __main__._display.drawExpandedCells(
              self._visitedlist)  # @UndefinedVariable

    return isGoal

  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.

     As noted in search.py:
         For a given state, this should return a list of triples,
     (successor, action, stepCost), where 'successor' is a
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental
     cost of expanding to that successor
    """

    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x, y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append((nextState, action, cost))

    # Bookkeeping for display purposes
    self._expanded += 1
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)

    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    if actions == None:
      return 999999
    x, y = self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += self.costFn((x, y))
    return cost