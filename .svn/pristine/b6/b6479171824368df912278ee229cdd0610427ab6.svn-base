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

import util
import os
import heapq


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
  w = Directions.WEST
  return [s, s, w, s, w, w, s, w]


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
