from random import randint
import numpy as np
import pandas as pd
import scipy

import sys
import utils
import copula
import warnings
import scipy.optimize as optimize
from scipy.integrate import quad
from statsmodels.nonparametric.kde import KDEUnivariate


import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
c_map = {0:'clayton', 1:'frank', 2:'gumbel'}
eps = np.finfo(np.float32).eps



class CopulaModel(object):
	"""This class instantiates a Copula Model from a dataset.
	Attributes:
		data_path: A string indicating the directory of the training dataset
		meta_path: A meta file specifies information for each column
		model_data: A dataframe storing training data
		u_type: methods for estimating density, can be 'kde' or 'gaussian'
		n_var: number of variables

		param: A list of parameters fitted to the data for each variable
		cdfs: A list of cdf function fitted to the data for each variable
		ppfs: A list of inverse cdf function fitting to the data for each variable
		u_matrix: A matrix represents the univariate distribution of size m*n,
			where m is the number of data points, and n is number of variables
	"""
	def __init__(self,data_path,utype,meta_path=None):
		self.data_path = data_path
		self.meta_path = meta_path
		self.model_data = pd.read_csv(data_path,sep=',', index_col=False,
			na_values=['NaN', 'nan', 'NULL', 'null'], low_memory=False)
		#for cancer data, drop id column
		# self.model_data.drop(self.model_data.columns[0], axis=1, inplace=True)
		# self.model_data.drop(self.model_data.columns[len(self.model_data.columns)-1], axis=1, inplace=True)
		self.u_type = utype
		self.n_sample = self.model_data.shape[0]
		self.n_var = self.model_data.shape[1]
		print('size of the data is:{0},{1}'.format(self.n_sample,self.n_var))

		#transform copulas into its univariate
		self.cdfs,self.u_matrix,self.ppfs= self._preprocessing(self.model_data)
		print('finish preprocessing steps')
		#information about the copula model
		self.tau_mat = self.model_data.corr(method='kendall').as_matrix()
		self.model = None
		self.param = None
		self._fit_model()


	def _preprocessing(self,data):
		"""Preprocessing steps for the dataframe before building copulas.

		Retrieve meta files, add noise for integer columns and
		compute the cdf,ppf for each column and transform data in cdfs.

		Returns:
		cdfs: list of cdf function generator for each column
		ppfs: list of ppf function generator for each column
		unis: np matrix of data after applying cdf function to each column
		"""
		cdfs = []
		ppfs = []
		unis = np.empty([data.shape[0],data.shape[1]])
		count = 0

		for col in data:
			#if string,map to categorical
			if isinstance(data[col][0], str):
				d = dict([(y,x+1) for x,y in enumerate(sorted(set(data[col])))])
				data[col]=[d[x] for x in data[col]]
				noise = np.random.normal(0,0.01,self.n_sample)
				data[col]=noise+data[col]
			kde = KDEUnivariate(data[col])
			kde.fit()
			dist = utils.Distribution(column=data[col],summary={'name':'kde','values':None})
			dist.name=self.u_type
			cdf = dist.cdf
			ppf = dist.ppf
			ppfs.append(ppf)
			cdfs.append(cdf)
			# unis[:,count]=[cdf(x) for x in list(data[col].values)]
			# unis[:,count]=cdf

			count+=1
		return cdfs,unis,ppfs


	def _fit_model(self):
		"""Fit copula model to the data
		Returns:
		self.param: param of copula family, tree_data if model is a vine
		"""
		vine = RVine(self.u_matrix,self.tau_mat,self.ppfs)
		self.model = vine
		self.param = vine.vine_model


	def sampling(self,n,plot=False,out_dir=None):
		sampled = np.zeros([n,self.n_var])
		for i in range(n):
			x = self.model._sampling(n)
			sampled[i,:]=x
		if plot:
			plt.scatter(self.model_data.ix[:, 0],self.model_data.ix[:, 1],c='green')
			plt.scatter(sampled[:,0],sampled[:,1],c='red')
			plt.show()
		if out_dir:
			np.savetxt(out_dir, sampled, delimiter=",")
		return sampled



class RVine(object):
	"""This class constructs a regular vine model consisting multiple levels of trees
	Attributes:
		u_matrix: matrix represents the univariate distribution of size m*n,
				where m is the number of data points, and n is number of variables
		tau_mat: matrix represents kendall tau matrix
		n_sample: number of samples in the dataset
		n_var: number of variables
		depth: depth of the vine model
		vine_model: array [level of tree] -> [tree]
	"""

	def __init__(self,u_mat,tau_mat,ppf,truncated=3):
		self.u_matrix= u_mat
		self.tau_mat = tau_mat

		self.truncated = truncated
		self.n_sample = self.u_matrix.shape[0]
		self.n_var = self.u_matrix.shape[1]
		self.depth = self.n_var - 1

		self.vine_model=[]
		self.ppfs = ppf
		self.train_vine()


	def train_vine(self):
		"""Train a vine model
		output: trees are stored in self.vine_model
		"""
		print('start building tree : 0')

		tree_1 = Tree(0,self.n_var,self.tau_mat,self.u_matrix)
		self.vine_model.append(tree_1)
		print('finish building tree : 0')
		tree_1.print_tree()
		for k in range(1,min(self.n_var-1,self.truncated)):
			'''get constraints from previous tree'''
			self.vine_model[k-1]._get_constraints()
			tau = self.vine_model[k-1]._get_tau()
			print('start building tree: {0}'.format(k))
			tree_k = Tree(k,self.n_var-k,tau,self.vine_model[k-1])
			self.vine_model.append(tree_k)
			print('finish building tree: {0}'.format(k))
			tree_k.print_tree()


	def _get_adjacent_matrix(self):
		"""Build adjacency matrix from the first tree
		"""
		first_tree = self.vine_model[0].edge_set
		n = len(first_tree)+1
		adj=np.zeros([n,n])
		for k in range(len(first_tree)):
			adj[first_tree[k].L,first_tree[k].R]=1
			adj[first_tree[k].R,first_tree[k].L]=1
		return adj


	def _sampling(self,n):
		first_tree = self.vine_model[0].edge_set
		"""generating samples from vine model"""
		unis = np.random.uniform(0,1,self.n_var)
		#randomly select a node to start with
		first_ind = randint(0,self.n_var-1)
		adj = self._get_adjacent_matrix()
		visited = []
		unvisited = set(range(self.n_var))
		explore = []
		explore.insert(0,first_ind)
		itr = 0
		sampled = [0]*self.n_var
		while explore:
			current = explore.pop(0)
			# print('processing variable : {0}'.format(current))
			neighbors = np.where(adj[current,:]==1)[0].tolist()
			if itr==0:
				new_x = self.ppfs[current](unis[current])
			else:
				for i in range(itr-1,-1,-1):
					current_ind = -1
					if i>=self.truncated:
						continue
					current_tree=self.vine_model[i].edge_set
					# print('inside loop number: {0}'.format(i))
					#get index of edge to retrieve
					for edge in current_tree:
						if i==0:
							if (edge.L==current and edge.R==visited[0]) or (edge.R==current and edge.L==visited[0]):
								current_ind = edge.index
								break
						else:
							if edge.L==current or edge.R==current:
								condition = set(edge.D)
								condition.add(edge.L)
								condition.add(edge.R)
								visit_set = set(visited).add(current)
								if condition.issubset(visited):
									current_ind = edge.index
								break
					if current_ind != -1:
						#the node is not indepedent contional on visited node
						copula_type = current_tree[current_ind].name
						copula_para = current_tree[current_ind].param
						cop = copula.Copula(1,1,theta=copula_para,cname=c_map[copula_type],dev=True)
						#start with last level
						if i==itr-1:
							tmp = optimize.fminbound(cop.derivative,eps,1.0,args=(unis[visited[0]],copula_para,unis[current]))
						else:
							tmp = optimize.fminbound(cop.derivative,eps,1.0,args=(unis[visited[0]],copula_para,tmp))
						mp=min(max(tmp,eps),0.99)
				new_x = self.ppfs[current](tmp)
			# print(new_x)
			sampled[current]=new_x
			for s in neighbors:
				if s in visited:
					continue
				else:
					explore.insert(0,s)
			itr+=1
			visited.insert(0,current)
		return sampled




class Tree():
	"""instantiate a single tree in the vine model
	:param k: level of tree
	:param prev_T: tree model of previous level
	:param tree_data: current tree model
	:param new_U: conditional cdfs for next level tree
	"""

	def __init__(self,k,n,tau_mat,prev_T):
		# super(Tree,self).__init__(copula, y_ind)
		self.level = k+1
		self.prev_T = prev_T
		self.n_nodes = n
		self.edge_set = []
		self.tau_mat = tau_mat
		if self.level == 1 :
			self.u_matrix = prev_T
			self._build_first_tree()
			# self.print_tree()
		else:
			# self.u_matrix = prev_T.u_matrix
			self._build_kth_tree()
		self._data4next_T()

	def identify_eds_ing(self,e1,e2):
		"""find nodes connecting adjacent edges
		:param e1: pair of nodes representing edge1
		:param e2: pair of nodes representing edge2
		:output ing: nodes connecting e1 and e2
		:output n1,n2: the other node of e1 and e2 respectively
		"""
		A = set([e1.L,e1.R])
		A.update(e1.D)
		B = set([e2.L,e2.R])
		B.update(e2.D)
		D = list(A&B)
		left = list(A^B)[0]
		right = list(A^B)[1]
		return left,right,D

	def check_adjacency(self,e1,e2):
		"""check if two edges are adjacent"""
		return (e1.L==e2.L or e1.L==e2.R or e1.R==e2.L or e1.R==e2.R)

	def check_contraint(self,e1,e2):
		full_node = set([e1.L,e1.R,e2.L,e2.R])
		full_node.update(e1.D)
		full_node.update(e2.D)
		return (len(full_node)==(self.level+1))

	def _get_constraints(self):
		"""get neighboring edges
		"""
		for k in range(len(self.edge_set)):
			for i in range(len(self.edge_set)):
				#add to constriants if i shared an edge with k
				if k!=i and self.check_adjacency(self.edge_set[k],self.edge_set[i]):
					self.edge_set[k].neighbors.append(i)


	def _get_tau(self):
		"""Get tau matrix for adjacent pairs
		:param tree: a tree instance
		:param ctr: map of edge->adjacent edges
		"""
		tau = np.empty([len(self.edge_set),len(self.edge_set)])
		for i in range(len(self.edge_set)):
			for j in self.edge_set[i].neighbors:
				# ed1,ed2,ing = tree.identify_eds_ing(self.edge_set[i],self.edge_set[j])
				edge = self.edge_set[i].parent
				l_p = edge[0]
				r_p = edge[1]
				if self.level == 1:
					U1,U2 = self.u_matrix[:,l_p],self.u_matrix[:,r_p]
				else:
					U1,U2 = self.prev_T.edge_set[l_p].U,self.prev_T.edge_set[r_p].U
				tau[i,j],pvalue = scipy.stats.kendalltau(U1,U2)
		return tau


	def _build_first_tree(self):
		"""build the first tree with n-1 variable
		"""
		tau_mat = self.tau_mat
        #Prim's algorithm
		neg_tau = -1.0*abs(tau_mat)
		X=set()
		X.add(0)
		itr=0
		while len(X)!=self.n_nodes:
			adj_set=set()
			for x in X:
				for k in range(self.n_nodes):
					if k not in X and k!=x:
						adj_set.add((x,k))
			#find edge with maximum
			edge = sorted(adj_set, key=lambda e:neg_tau[e[0]][e[1]])[0]
			cop = copula.Copula(self.u_matrix[:,edge[0]],self.u_matrix[:,edge[1]])
			name,param=cop.select_copula(cop.U,cop.V)
			new_edge = Edge(itr,edge[0],edge[1],tau_mat[edge[0],edge[1]],name,param)
			new_edge.parent.append(edge[0])
			new_edge.parent.append(edge[1])
			self.edge_set.append(new_edge)
			X.add(edge[1])
			itr+=1


	def _build_kth_tree(self):
		"""build tree for level k
		"""
		neg_tau = -abs(self.tau_mat)
		visited=set()
		unvisited = set(range(self.n_nodes))
		visited.add(0) #index from previous edge set
		unvisited.remove(0)
		itr=0
		while len(visited)!=self.n_nodes:
			adj_set=set()
			for x in visited:
				for k in range(self.n_nodes):
					if k not in visited and k!=x:
						#check if (x,k) is a valid edge in the vine
						if self.check_contraint(self.prev_T.edge_set[x],self.prev_T.edge_set[k]):
							adj_set.add((x,k))
			#find edge with maximum tau
			# print('processing edge:{0}'.format(x))
			print(itr)
			print(adj_set)
			if len(list(adj_set)) == 0:
				visited.add(list(unvisited)[0])
				continue
			edge = sorted(adj_set, key=lambda e:neg_tau[e[0]][e[1]])[0]

			[ed1,ed2,ing]=self.identify_eds_ing(self.prev_T.edge_set[edge[0]],self.prev_T.edge_set[edge[1]])
			# U1 = self.u_matrix[ed1,ing]
			# U2 = self.u_matrix[ed2,ing]
			l_p = edge[0]
			r_p = edge[1]
			U1,U2 = self.prev_T.edge_set[l_p].U,self.prev_T.edge_set[r_p].U
			cop = copula.Copula(U1,U2,self.tau_mat[edge[0],edge[1]])
			name,param=cop.select_copula(cop.U,cop.V)
			new_edge = Edge(itr,ed1,ed2,self.tau_mat[edge[0],edge[1]],name,param)
			new_edge.D = ing
			new_edge.parent.append(edge[0])
			new_edge.parent.append(edge[1])
			self.edge_set.append(new_edge)
			visited.add(edge[1])
			unvisited.remove(edge[1])
			itr+=1



	def _data4next_T(self):
		"""
		prepare conditional U matrix for next tree
		"""
		# U = np.empty([self.n_nodes,self.n_nodes],dtype=object)
		edge_set = self.edge_set
		for k in range(len(edge_set)):
			edge = edge_set[k]
			copula_name = c_map[edge.name]
			copula_para = edge.param
			if self.level == 1:
				U1,U2 = self.u_matrix[:,edge.L],self.u_matrix[:,edge.R]
			else:
				prev_T = self.prev_T.edge_set
				l_p = edge.parent[0]
				r_p = edge.parent[1]
				U1,U2 = prev_T[l_p].U,prev_T[r_p].U
			'''compute conditional cdfs C(i|j) = dC(i,j)/duj and dC(i,j)/dui'''
			U1=[x for x in U1 if x is not None]
			U2=[x for x in U2 if x is not None]

			c1= copula.Copula(U2,U1,theta=copula_para,cname=copula_name,dev=True)
			U1givenU2 = c1.derivative(U2,U1,copula_para)
			U2givenU1 = c1.derivative(U1,U2,copula_para)

			'''correction of 0 or 1'''
			U1givenU2[U1givenU2==0],U2givenU1[U2givenU1==0]=eps,eps
			U1givenU2[U1givenU2==1],U2givenU1[U2givenU1==1]=1-eps,1-eps
			edge.U = U1givenU2


	def _likehood_T(self,U):
		"""Compute likelihood of the tree given an U matrix
		"""
		# newU = np.
		newU = np.empty([self.vine.n_var,self.vine.n_var])
		tree = self.tree_data
		values = np.zeros([1,tree.shape[0]])
		for i in range(tree.shape[0]):
			cname = self.vine.c_map[int(tree[i,4])]
			v1 = int(tree[i,1])
			v2 = int(tree[i,2])
			copula_para = tree[i,5]
			if self.level == 1:
				U_arr = np.array([U[v1]])
				V_arr = np.array([U[v2]])
				cop = copula.Copula(U_arr,V_arr,theta=copula_para,cname=cname,dev=True)
				values[0,i]=cop.pdf(U_arr,V_arr,copula_para)
				U1givenU2 = cop.derivative(V_arr,U_arr,copula_para)
				U2givenU1 = cop.derivative(U_arr,V_arr,copula_para)
			else:
				v1 = int(tree[i,6])
				v2 = int(tree[i,7])
				joint = int(tree[i,8])
				U1 = np.array([U[v1,joint]])
				U2 = np.array([U[v2,joint]])
				cop = copula.Copula(U1,U2,theta=copula_para,cname=cname,dev=True)
				values[0,i] = cop.pdf(U1,U2,theta=copula_para)
				U1givenU2 = cop.derivative(U2,U1,copula_para)
				U2givenU1 = cop.derivative(U1,U2,copula_para)
			newU[v1,v2]=U1givenU2
			newU[v2,v1]=U2givenU1
		# print(values)
		value = np.sum(np.log(values))
		return newU,value

	def print_tree(self):
		for e in self.edge_set:
			print(e.L,e.R,e.D,e.parent)



class Edge(object):
	def __init__(self,index,left,right,tau,copula_name,copula_para):
		self.index = index #index of the edge in the current tree
		self.L = left  #left_node index
		self.R = right #right_node index
		self.D = [] #dependence_set
		self.parent = [] #indices of parent edges in the previous tree
		self.tau = tau   #correlation of the edge
		self.name = copula_name
		self.param = copula_para
		self.U = None
		self.likelihood = None
		self.neighbors = []



if __name__ == '__main__':
	print('start building')
	model = CopulaModel('data/K9.data','kde','rvine')
	sample=model.sampling(16772,plot=False,out_dir='experiments/mutant_synthetic.csv')
	# print(sample)

	# model.predict('test_2.csv')
