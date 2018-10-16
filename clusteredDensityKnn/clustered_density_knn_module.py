import numpy as np
import pandas
import os
from sklearn import neighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KernelDensity

class ClusteredDensityKnnModule(object):
	data_set_samples = []
	data_set_labels = []
	test_data_set_samples = []
	test_data_set_labels = []
	distance = []
	clusters = []
	clusters_training = []
	indices = []
	k_neighbors = 1
	n_clusters = 2

	def __init__(self):
		print("init knn module")

	def run(self):

		
		#print(self.data_set_samples)
		#print(self.data_set_labels)
		#exit()
		'''self.findClusters(self.data_set_samples)

		self.findNearestNeighbors(self.data_set_samples)
		#print(self.distance)
		#print(self.distance[5])
		#print(self.data_set_samples)
		#print(self.data_set_labels)'''
		self.getDistanceDensity(self.data_set_samples)

		print((self.distance))
		exit()

		clf = neighbors.KNeighborsClassifier(self.k_neighbors)
		clf.fit(self.distance, self.data_set_labels)


		print("INICIANDO FASE DE TESTE...")
		self.findClusters(self.test_data_set_samples, self.data_set_samples)
		self.findNearestNeighbors(self.test_data_set_samples)
		print((self.distance))


		
		predictions = clf.predict(self.distance)
		print(predictions)
		print((self.distance))
		#exit()
		return predictions

	def findClusters(self, data_set,  data_set_training=None):
		self.clusters = []
		self.clusters_training = []
		self.indices = []
		self.distance = []
		self.labels = []

		kde = neighbors.KernelDensity(bandwidth=0.75).fit(data_set)
		
		print(data_set)
		if (data_set_training == None):
			data_set_training = data_set
		else:
			mergedlist = []
			mergedlist.extend(data_set)
			mergedlist.extend(data_set_training)
			data_set_training = mergedlist
		
		#print("data_set_trainging:")
		#print(data_set_training)
		#print(len(data_set_training))
		#print(len(data_set_training[0]))


		kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=0).fit(data_set_training)
		#predicao = kmeans.predict(data_set_training)

		#print("Centros de cluster")
		#print(kmeans.cluster_centers_)
		#exit()

		distance_clusters = kmeans.fit_transform(data_set_training)
		#print(distance_clusters)
		#print(len(distance_clusters))
		#print(len(distance_clusters[0]))
		#print(len(data_set))
		#distance_clusters_data_set = distance_clusters[0:len(data_set)]
		
		#print(distance_clusters)
		#exit()

		#print("Preparando clusters...")
		for i in range(0, len(kmeans.cluster_centers_)):
			self.clusters.append([])
			self.clusters_training.append([])
			self.indices.append([])

		for i in range(0, len(data_set_training)):
			self.clusters_training[kmeans.labels_[i]].append(data_set_training[i]) 

		#print(len(data_set))
		#print(len(distance_clusters))
		#print(len(kmeans.labels_))
		predicao = kmeans.predict(data_set_training)

		for i in range(0,len(data_set)):
			dist = 0
			print(("exemplo: ", i+1))
			for j in distance_clusters[i]:
				#print("Distancia para cluster: ", j )
				dist+= j


				#print(j)
			#self.distance.append(dist)
			self.distance.append([dist])
			
			'''print("dATASET: ", data_set[i])
			print("data set training: " ,data_set_training[i])
			print(kmeans.labels_)
			print(predicao[i])
			print(dist)
			print("------------")'''
			self.indices[kmeans.labels_[i]].append(i)
			self.clusters[kmeans.labels_[i]].append(data_set[i]) 

			#print(self.clusters)
			density = kde.score_samples(data_set[i])
			self.distance[i].append(density[0])

		#print(self.distance)
		#exit()
		return distance_clusters

	def findNearestNeighbors(self, data_set):
		print("Procurando distancia para vizinho mais proximo...")
		#print(len(self.clusters))
		#print(len(self.clusters[1]))
		#exit()

		#print(self.distance[0])

		for i in range(0, len(self.clusters)):
			if(len(self.clusters_training[i]) > 1):
				clf = neighbors.NearestNeighbors(n_neighbors=2)
				for j in  range(0, len(self.clusters[i])):
					clf.fit(self.clusters_training[i])
					#print(len(self.clusters_training[i]))
					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
					self.distance[self.indices[i][j]]+= neighbor[0][0][1]
					'''print("indice: ", self.indices[i][j])
					print("Exemplo: ", self.clusters[i][j])
					print("original", data_set[self.indices[i][j]])
					print("distancia neighbor", neighbor[0][0][1])
					print("Distancia: ", self.distance[self.indices[i][j]])'''
					print((self.indices[i][j]))

			else:
				clf = neighbors.NearestNeighbors(n_neighbors=1)

				for j in  range(0, len(self.clusters[i])):
					clf.fit(self.clusters_training[i])
					#print(len(self.clusters_training[i]))
					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
					self.distance[self.indices[i][j]]+= neighbor[0][0][0]
					'''print("indice: ", self.indices[i][j])
					print("Exemplo: ", self.clusters[i][j])
					print("original", data_set[self.indices[i][j]])
					print("distancia neighbor", neighbor[0][0][0])
					print("Distancia: ", self.distance[self.indices[i][j]])
					print(self.indices[i][j])'''

		#exit()			
		#print(self.distance)

	def getDistanceDensity(self, data_set):
		self.distance = []
		kde = neighbors.KernelDensity(kernel = 'linear',bandwidth=0.75).fit(data_set)

		for i in range(0,len(data_set)):
			density = kde.score_samples(data_set[i])
			self.distance.append(density)

	def setDataSetClustering(self, data_set):
		self.data_set_samples = data_set.values[:,0:(len(data_set.values[0])-1)]
		print((self.data_set_samples))

	def setDataSet(self, data_set):
		print(data_set)
		self.data_set_samples = data_set.values[:,0:(len(data_set.values[0])-2)]
		self.data_set_labels = data_set.values[:,(len(data_set.values[0])-2)]
		print((self.data_set_samples))
		#exit()
		#print(self.data_set_samples)
		#print(self.data_set_labels)


	
	def setTestDataSet(self, test_data_set):
		print("1")
		print(test_data_set)
		print("2")
		self.test_data_set_samples = test_data_set.values[:,0:(len(test_data_set.values[0])-2)]
		#self.test_data_set_samples = test_data_set.values[:,0:(len(test_data_set.values[0])-2)]
		self.test_data_set_labels = test_data_set.values[:,(len(test_data_set.values[0])-2)]		
		print("a")
		print((self.test_data_set_samples))
		print("b")
		print((self.test_data_set_labels))	
		print("c")
		#exit()
	

	def setKNeighbors(self, k_neighbors):
		self.k_neighbors = k_neighbors

	def getKNeighbors(self):
		return self.k_neighbors

	def setNClusters(self, n_clusters):
		self.n_clusters = n_clusters

	def getNClusters(self):
		return self.n_clusters

	

	'''def findNearestNeighbors(self, data_set, labels):
		print("Procurando distancia para vizinho mais proximo...")
		print(len(self.clusters))
		print(len(self.clusters[1]))
	 	for i in range(0, len(self.clusters)):
	 		clf = neighbors.KNeighborsClassifier()
			#clf = neighbors.NearestNeighbors(n_neighbors=2)
			clf.fit(self.clusters_training[i], self.labels[i])
			a = clf.kneighbors(n_neighbors=1)
			print(a) 

			exit()
			print("Cluster:")
			print(self.clusters_training[i])
			#clf.fit(self.clusters_training[i])
			for j in  range(0, len(self.clusters[i])):
				print("Exemplo: ", self.clusters[i][j])
				print("indice: ", self.indices[i][j])
				print("original", data_set[self.indices[i][j]])
				print("label: ", self.labels[i][j])
				print("label original: ", labels[self.indices[i][j]] )
				#neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
				#print("distancia neighbor", neighbor[0][0][1])
				print(self.indices[i][j])
				#self.distance[self.indices[i][j]]+= neighbor[0][0][1]
		exit()
		#print(self.distance)


	def findClusters(self, data_set, data_set_labels,  data_set_training=None):
		self.clusters = []
		self.clusters_training = []
		self.indices = []
		self.distance = []
		self.labels = []
		
		if (data_set_training == None):
			data_set_training = data_set
		else:
			mergedlist = []
			mergedlist.extend(data_set)
			mergedlist.extend(data_set_training)
			data_set_training = mergedlist
			print(data_set_training)

		kmeans = MiniBatchKMeans(n_clusters=2, random_state=0).fit(data_set_training)
		#predicao = kmeans.predict(data_set_training)

		print("Centros de cluster")
		print(kmeans.cluster_centers_)
		distance_clusters = kmeans.fit_transform(data_set_training)
		print(distance_clusters)
		print(len(distance_clusters))
		print(len(data_set))
		#distance_clusters_data_set = distance_clusters[0:len(data_set)]
		print(distance_clusters)

		print("Preparando clusters...")
		for i in range(0, len(kmeans.cluster_centers_)):
			self.clusters.append([])
			self.labels.append([])
			self.clusters_training.append([])
			self.indices.append([])

		for i in range(0, len(data_set_training)):
			self.clusters_training[kmeans.labels_[i]].append(data_set_training[i]) 

		print(len(data_set))
		print(len(distance_clusters))
		print(len(kmeans.labels_))
		predicao = kmeans.predict(data_set_training)

		for i in range(0,len(data_set)):
			dist = 0
			for j in distance_clusters[i]:
				dist+= j 
				#print(j)
			#self.distance.append(dist)
			self.distance.append([dist])
			#print(dist)
			#print(kmeans.labels_[i])
			#print(predicao[i])
			#print("------------")
			self.indices[kmeans.labels_[i]].append(i)
			self.clusters[kmeans.labels_[i]].append(data_set[i]) 
			self.labels[kmeans.labels_[i]].append(data_set_labels[i]) 

		#exit()
		#print(self.distance)
		#print(self.clusters)

		return distance_clusters

	def findNearestNeighbors(self, data_set, labels=None):
		print("Procurando distancia para vizinho mais proximo...")
		print(len(self.clusters))
		print(len(self.clusters[1]))

		if(labels == None):
			for i in range(0, len(self.clusters)):
				clf = neighbors.NearestNeighbors(n_neighbors=2)
				for j in  range(0, len(self.clusters[i])):
					clf.fit(self.clusters_training[i])
					neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
					print("distancia neighbor", neighbor[0][0][1])
					self.distance[self.indices[i][j]]+= neighbor[0][0][1]
		else:
		 	for i in range(0, len(self.clusters)):
		 		clf = neighbors.KNeighborsClassifier()
			
				clf.fit(self.clusters_training[i], self.labels[i])
				neighbor = clf.kneighbors(n_neighbors=1)

				#exit()
				print("Cluster:")
				print(self.clusters_training[i])
				#clf.fit(self.clusters_training[i])
				for j in  range(0, len(self.clusters[i])):
					print("Exemplo: ", self.clusters[i][j])
					print("indice: ", self.indices[i][j])
					print("original", data_set[self.indices[i][j]])
					print("label: ", self.labels[i][j])
					print("label original: ", labels[self.indices[i][j]] )
					#neighbor = clf.kneighbors(self.clusters[i][j], return_distance=True)
					#print("distancia neighbor", neighbor[0][0][1])
					print(self.indices[i][j])
					print("vizinho: ",neighbor[0][j])
					#exit()
					self.distance[self.indices[i][j]]+= neighbor[0][j]
		#exit()
		print(self.distance)

	'''