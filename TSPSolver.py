#!/usr/bin/python3

#from which_pyqt import PYQT_VER
#if PYQT_VER == 'PYQT5':
# 	from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
# 	from PyQt4.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT6':
from PyQt6.QtCore import QLineF, QPointF
# else:
# 	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
from heapq import *
import itertools
from collections import deque
from random import randrange


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = None
		foundTour = False
		start_time = time.time()

		start_city = 0
		# initialize dist array with distance values
		dist = np.empty((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				dist[i,j] = cities[i].costTo(cities[j])
		
		while not foundTour:
			curr = dist.copy()
			# initialize path
			path = [start_city]
			curr[:,start_city] = np.inf
			while len(path) < ncities:
				# find shortest adjacent path
				start = path[-1]
				target = np.argmin(curr[start,:])
				path.append(target)
				# cross off newly unavailable paths
				curr[start,:] = np.inf
				curr[:,target] = np.inf
				curr[target,start] = np.inf
			
			bssf = TSPSolution([cities[i] for i in path])
			if bssf.cost < np.inf:
				foundTour = True
			else:
				start_city += 1
				if start_city > ncities:
					break

		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def greedy_all( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = None
		start_time = time.time()

		# initialize dist array with distance values
		dist = np.empty((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				dist[i,j] = cities[i].costTo(cities[j])
		
		solns = []
		for start_city in range(ncities):
			curr = dist.copy()
			# initialize path
			path = [start_city]
			curr[:,start_city] = np.inf
			while len(path) < ncities:
				# find shortest adjacent path
				start = path[-1]
				target = np.argmin(curr[start,:])
				path.append(target)
				# cross off newly unavailable paths
				curr[start,:] = np.inf
				curr[:,target] = np.inf
				curr[target,start] = np.inf
			
			solns.append(TSPSolution([cities[i] for i in path]))
		
		bssf = min(solns, key=lambda s: s.cost)

		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	def bi_greedy_all( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = None
		start_time = time.time()

		# initialize dist array with distance values
		dist = np.empty((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				dist[i,j] = cities[i].costTo(cities[j])
		
		solns = []
		for start_city in range(ncities):
			curr = dist.copy()
			# initialize path
			path = deque([start_city])
			curr[:,start_city] = np.inf
			while len(path) < ncities:
				# find shortest adjacent path
				bstart = path[-1]
				fstart = path[0]
				ftarget = np.argmin(curr[:,fstart])
				btarget = np.argmin(curr[bstart,:])
				
				# cross off newly unavailable paths
				if curr[ftarget,fstart] < curr[bstart,btarget]:
					path.appendleft(ftarget)
					curr[ftarget,:] = np.inf
					curr[:,fstart] = np.inf
					curr[fstart,ftarget] = np.inf
				else:
					path.append(btarget)
					curr[bstart,:] = np.inf
					curr[:,btarget] = np.inf
					curr[btarget,bstart] = np.inf
			
			solns.append(TSPSolution([cities[i] for i in path]))
		
		bssf = min(solns, key=lambda s: s.cost)

		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def k2OptRandom( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		start_time = time.time()
		bssf = self.greedy_all()['soln']

		# initialize dist array with distance values
		dist = np.empty((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				dist[i,j] = cities[i].costTo(cities[j])

		while time.time()-start_time < time_allowance:
			c2 = randrange(ncities)
			d2 = randrange(ncities)
			if c2 > d2:
				c2,d2 = d2,c2
			c1 = c2 - 1
			d1 = d2 - 1

			newRoute = TSPSolution(bssf.route[0:c1] + [bssf.route[c1]] + bssf.route[d1:c1:-1] + bssf.route[d2:])
			if newRoute.cost < bssf.cost:
				bssf = newRoute
				count += 1
		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def k2OptOrdered( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		start_time = time.time()
		bssf = self.greedy_all()['soln']

		while time.time() - start_time < time_allowance:
			for i in range(-1, ncities - 1):
				for j in range(i + 1, ncities):
					newRoute = TSPSolution(bssf.route[0:i] + [bssf.route[i]] + bssf.route[j:i:-1] + bssf.route[j+1:])
					if newRoute.cost < bssf.cost:
						bssf = newRoute
						count += 1
						break
				else:
					continue
				break
			else:
				break

		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	def k3OptOrdered( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		start_time = time.time()
		bssf = self.greedy_all()['soln']

		while time.time() - start_time < time_allowance:
			for i in range(-1, ncities - 2):
				for j in range(i + 1, ncities - 1):
					for k in range(j + 1, ncities):
						r = bssf.route
						routes = [
							# 0 inverted
							bssf,
							# 1 inverted
							TSPSolution(r[0:i] + r[j-1:i:-1] + [r[i]] + r[j:]),
							TSPSolution(r[0:j] + r[k-1:j:-1] + [r[j]] + r[k:]),
							TSPSolution(r[0:i] + r[k-1:i:-1] + [r[i]] + r[k:]),
							# 2 inverted
							TSPSolution(r[0:i] + r[k-1:j:-1] + [r[j]] + r[i:j] + r[k:]),
							TSPSolution(r[0:i] + r[j:k] + r[j-1:i:-1] + [r[i]]  + r[k:]),
							TSPSolution(r[0:i] + r[j-1:i:-1] + [r[i]] + r[k-1:j:-1] + [r[j]] + r[k:]),
							# 3 inverted
							TSPSolution(r[0:i] + r[j:k] + r[i:j] + r[k:]),
						]

						newRoute = min(routes, key=lambda r: r.cost)

						if newRoute.cost < bssf.cost:
							bssf = newRoute
							count += 1
							break
					else:
						continue
					break
				else:
					continue
				break
			else:
				break

		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		def updateRCM(dist, L=0, targetCity=0, backtrace=[]):
			"""Reduce the given matrix (0 in each row/column) and update the L-value accordingly.
			   Duplicate the backtrace array and append the new target city"""
			for i in range(len(dist)): # rows
				m = np.amin(dist[i,:])
				if m == np.inf: continue

				dist[i] -= m
				L += m
			
			for j in range(len(dist)): # columns
				m = np.amin(dist[:,j])
				if m == np.inf: continue

				dist[:,j] -= m
				L += m
			
			newBacktrace = backtrace.copy()
			newBacktrace.append(targetCity)
			return L, dist, newBacktrace
		
		def pruneQ(Q, L, qCounter):
			oldQlen = len(Q)
			Q = [e for e in Q if e[0] < L]
			heapify(Q)
			qCounter += (oldQlen - len(Q))
		
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		maxQSize = 0
		stateCount = 1
		pruneCount = 1
		# bssf = self.defaultRandomTour()['soln']
		start_time = time.time()
		bssf = self.greedy_all()['soln']
		
		# initialize dist array with distance values
		dist = np.empty((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				dist[i,j] = cities[i].costTo(cities[j])

		# use this generator to break ties in the heap Q, ensuring each entry has a unique key
		tiebreaker = itertools.count()
		
		# initialize Q with the reduced dist and L value
		L, dist, backtrace = updateRCM(dist)
		# print(dist)
		Q = [(L, next(tiebreaker), dist, backtrace)]
		heapify(Q)

		# iterate while there are nodes in Q and we have time
		while len(Q) and (time.time() - start_time < time_allowance):
			L, _, curr, backtrace = heappop(Q)
			startCity = backtrace[-1]
			
			for targetCity in range(ncities): # iterate over destinations
				stateCount += 1
				pathCost = curr[startCity,targetCity]
				if pathCost == np.inf:
					pruneCount += 1
					continue # don't process if there is no path
				
				# copy array and cross off newly unavailable paths
				arr = curr.copy()
				arr[startCity,:] = np.inf
				arr[:,targetCity] = np.inf
				arr[targetCity,startCity] = np.inf

				# reduce arr and get a new L and arr value 
				newL, newArr, newBacktrace = updateRCM(arr, L, targetCity, backtrace)
				newL += pathCost
				# don't continue if not better than bssf
				if newL >= bssf.cost:
					pruneCount += 1
					continue
				# if this is a leaf
				if len(newBacktrace) == ncities:
					# print(newBacktrace)
					bssf = TSPSolution([cities[i] for i in newBacktrace])
					pruneQ(Q, newL, pruneCount)
					count += 1
				else: # otherwise push the new array
					heappush(Q, (newL, next(tiebreaker), newArr, newBacktrace))
			# update maxQSize
			maxQSize = max(maxQSize, len(Q))
		
		pruneCount += len(Q)
		
		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxQSize
		results['total'] = stateCount
		results['pruned'] = pruneCount
		print(results)
		return results


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass
