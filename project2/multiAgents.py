# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project.  You are free to use and extend these projects for educational
# purposes.  The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from util import manhattanDistance
from game import Directions
import random
import util
from mypy import MinimaxDecision, AlphaBetaMiniMaxDecsion, ExpectedMiniMaxDecision, mazeDistance

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action)
                                      for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(
        len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

    "Add more of your code here if you want to"

    # print legalMoves[chosenIndex]
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function"
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)

    oldPos = currentGameState.getPacmanPosition()
    newPos = successorGameState.getPacmanPosition()

    oldFood = currentGameState.getFood()
    newFood = successorGameState.getFood()

    oldGhostStates = currentGameState.getGhostStates()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newGhostDistances = [manhattanDistance(
        newPos, ghostState.getPosition()) for ghostState in newGhostStates]
    oldGhostDistances = [manhattanDistance(
        oldPos, ghostState.getPosition()) for ghostState in oldGhostStates]

    newFoodList = newFood.asList()
    newFoodDistances = [manhattanDistance(
        newPos, food) for food in newFoodList]
    # print newGhostDistances
    # print successorGameState
    # print newPos, newGhostStates[0].getPosition()

    score = 0
    if(len(newFoodDistances) == 0):
        score = 100000000 * successorGameState.getScore()
    elif(min(newGhostDistances) > 2):
        score = 100000000 * successorGameState.getScore() + (100 / (min(newFoodDistances) + 1)
                                                        ) - (1 / (min(newGhostDistances) + 1))
        if(action == Directions.STOP):
            score -= 500
    else:
        score = 100 * (min(newGhostDistances) - min(oldGhostDistances))

    return score


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
    self.index = 0  # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    return MinimaxDecision(gameState, self.evaluationFunction, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    return AlphaBetaMiniMaxDecsion(gameState, self.evaluationFunction, self.depth)


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    def maxLevel(gameState, depth):
      currDepth = depth + 1
      if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
        return self.evaluationFunction(gameState)
      maxvalue = -999999
      actions = gameState.getLegalActions(0)
      totalmaxvalue = 0
      numberofactions = len(actions)
      for action in actions:
        successor = gameState.generateSuccessor(0, action)
        maxvalue = max(maxvalue, expectLevel(successor, currDepth, 1))
      return maxvalue

    def expectLevel(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(agentIndex)
      totalexpectedvalue = 0
      numberofactions = len(actions)
      for action in actions:
        successor = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == (gameState.getNumAgents() - 1):
          expectedvalue = maxLevel(successor, depth)
        else:
          expectedvalue = expectLevel(successor, depth, agentIndex+1)
        totalexpectedvalue = totalexpectedvalue + expectedvalue
      if numberofactions == 0:
        return 0
      return float(totalexpectedvalue)/float(numberofactions)

    actions = gameState.getLegalActions(0)
    currentScore = -999999
    returnAction = ''
    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      score = expectLevel(nextState, 0, 1)
      if score > currentScore:
        returnAction = action
        currentScore = score
    return returnAction


def betterEvaluationFunction(currentGameState):
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  """ Manhattan distance to the foods from the current state """
  foodList = newFood.asList()
  from util import manhattanDistance
  foodDistance = [0]
  for pos in foodList:
    foodDistance.append(manhattanDistance(newPos, pos))

  """ Manhattan distance to each ghost from the current state"""
  ghostPos = []
  for ghost in newGhostStates:
    ghostPos.append(ghost.getPosition())
  ghostDistance = [0]
  for pos in ghostPos:
    ghostDistance.append(manhattanDistance(newPos, pos))

  numberofPowerPellets = len(currentGameState.getCapsules())

  score = 0
  numberOfNoFoods = len(newFood.asList(False))
  sumScaredTimes = sum(newScaredTimes)
  sumGhostDistance = sum(ghostDistance)
  reciprocalfoodDistance = 0
  if sum(foodDistance) > 0:
    reciprocalfoodDistance = 1.0 / sum(foodDistance)

  score += currentGameState.getScore() + reciprocalfoodDistance + numberOfNoFoods

  if sumScaredTimes > 0:
    score += sumScaredTimes + \
        (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
  else:
    score += sumGhostDistance + numberofPowerPellets
  return score


better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

