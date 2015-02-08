#!/usr/local/bin/python

print 'importing libs...'
import os
import math
import scipy.cluster
import scipy.spatial
import numpy
import random
import sklearn.ensemble
#import sklearn.linear_model
import sys

start_index = None
end_index = None
if len(sys.argv) == 2:
	start_index = int(sys.argv[1].split('-')[0])
	end_index = int(sys.argv[1].split('-')[1])
	print "start at index: " + str(start_index)
	print "end at index: " + str(end_index)

driver_dir = 'C:\\Users\\kuanlin\\Desktop\\Kaggle\\contest_data\\drivers\\'
result = 'C:\\Users\\kuanlin\\Desktop\\Kaggle\\contest_data\\clustering_with_boosting\\' + str(start_index) + '-' + str(end_index) + 'run_result.csv'
#clustering threshold
threshold = 0.4
#distribution sliding window
sliding_win = 10
#distribution percentiles
ptiles = [10,25,50,75,90]

max_pcnt_outlier_per_clst = 0.05
max_pcnt_outlier_per_driver = 0.1

def standardizeCols(input_matrix, inplace=False):
	col_vectors = []
	for i in xrange(len(input_matrix)):
		if i == 0:
			for val in input_matrix[i]:
				col_vectors.append([val])
		else:
			for j in xrange(len(input_matrix[i])):
				col_vectors[j].append(input_matrix[i][j])
	mean_std = []
	for v in col_vectors:
		mean_std.append((numpy.mean(v), numpy.std(v)))
	
	del(col_vectors)
	
	if inplace:
		output_matrix = input_matrix
	else:
		output_matrix = []
	
	for row in input_matrix:
		if not inplace:
			output_row = []
		for i in xrange(len(row)):
			if mean_std[i][1] != 0:
				if inplace:
					row[i] = (row[i] - mean_std[i][0])/mean_std[i][1]
				else:
					output_row.append((row[i] - mean_std[i][0])/mean_std[i][1])
		if not inplace:
			output_matrix.append(output_row)
	
	return output_matrix

def maxInterval(input_list, percentile):
	target_num = numpy.percentile(input_list, percentile)
	max_record = 0
	cur_record = 0
	for item in input_list:
		if item > target_num:
			cur_record += 1
			if cur_record > max_record:
				max_record = cur_record
		else:
			cur_record = 0
	return max_record
	
def appendTripFeatureVector(data_dir, filename, data_matrix):
	#print os.path.join(data_dir, filename)
	trip_vel_dist = []
	trip_acce_dist = []
	cosine_sim = []
	head_index = 0
	tail_index = -1
	prev_x = 0.0
	prev_y = 0.0
	prev_head_x = 0.0
	prev_head_y = 0.0
	cum_dist = 0.0
	prev_speed = 0.0
	win_size = 0.0
	
	total_dist = 0.0
	total_sec = 0
	max_speed = 0.0
	min_acce = 0.0
	max_acce = 0.0

	for line in open(os.path.join(data_dir, filename), 'r'):
		line = line.strip()
		if line == '':
			continue
		if line.upper().startswith('X,Y'):
			continue
		
		tail_index += 1
		lineArr = line.split(',')
		cur_x = float(lineArr[0])
		cur_y = float(lineArr[1])
		traveled_dist = math.sqrt(math.pow(cur_x - prev_x, 2) + math.pow(cur_y - prev_y, 2))
		total_dist += traveled_dist
		total_sec += 1
		
		if (tail_index - head_index) > 0 and (tail_index - head_index) < sliding_win:
			cum_dist += traveled_dist
			win_size += 1.0
		else:
			if win_size >= float(sliding_win) / 2.0:
				cur_speed = cum_dist / win_size
				trip_vel_dist.append(cur_speed)
				if cur_speed > max_speed:
					max_speed = cur_speed
				if head_index > 0:
					curr_acce = (cur_speed-prev_speed)/win_size
					trip_acce_dist.append(curr_acce)
					if (cur_x!= 0 or cur_y!=0) and (prev_head_x!=0 or prev_head_y!=0):
						cosine_sim.append(scipy.spatial.distance.cosine([cur_x, cur_y], [prev_head_x, prev_head_y]))
					if curr_acce < min_acce:
						min_acce = curr_acce
					if curr_acce > max_acce:
						max_acce = curr_acce
						
					prev_speed = cur_speed
					prev_head_x = cur_x
					prev_head_y = cur_y
		
			head_index = tail_index
			if head_index > 0:
				cum_dist = traveled_dist
				win_size = 1.0
		
		prev_x = cur_x
		prev_y = cur_y
		
	if win_size > 1.0 and win_size >= float(sliding_win) / 2.0:
		cur_speed = cum_dist / win_size
		trip_vel_dist.append(cur_speed)
		if cur_speed > max_speed:
			max_speed = cur_speed
		if head_index > 0:
			curr_acce = (cur_speed-prev_speed)/win_size
			trip_acce_dist.append(curr_acce)
			if (cur_x!= 0 or cur_y!=0) and (prev_head_x!=0 or prev_head_y!=0):
				cosine_sim.append(scipy.spatial.distance.cosine([cur_x, cur_y], [prev_head_x, prev_head_y]))
			if curr_acce < min_acce:
				min_acce = curr_acce
			if curr_acce > max_acce:
				max_acce = curr_acce
	
	total_traveled = math.sqrt(math.pow(0.0 - prev_x, 2) + math.pow(0.0 - prev_y, 2))
	data_matrix.append([numpy.percentile(trip_vel_dist, i) for i in ptiles] +
						[numpy.percentile(trip_acce_dist, i) for i in ptiles] +
						[numpy.percentile(cosine_sim, i) for i in ptiles] +
						[total_dist, max_speed, min_acce, max_acce, total_sec, total_traveled, total_traveled/total_dist, maxInterval(trip_vel_dist, 50), maxInterval(trip_vel_dist, 75)]
						)
	
def extractDriverFeatures(trip_data_dir):
	print "processing " + trip_data_dir
	data_matrix = []
	trip_list = []

	for filename in os.listdir(trip_data_dir):
		if not filename.upper().endswith(".CSV"):
			continue
		trip_id = filename.upper().replace('.CSV','')
		trip_list.append(trip_id)
		appendTripFeatureVector(trip_data_dir, filename, data_matrix)
		
	return (trip_list, data_matrix)

def makeLabelWithClst(cluster_result, max_outlier_per_cluster, max_outlier_per_driver):
	cls_map = {}
	for i in xrange(len(cluster_result)):
		cls_id = cluster_result[i]
		if cls_id in cls_map:
			cls_map[cls_id].append(i)
		else:
			cls_map[cls_id] = [i]
			
	outliers = []
	for cls_id in cls_map:
		if len(cls_map[cls_id]) <= max_outlier_per_cluster:
			outliers += cls_map[cls_id]
	
	while len(outliers) > max_outlier_per_driver:
		del outliers[random.randint(0, len(outliers)-1)]
	
	#print outliers
	return [0 if i in outliers else 1 for i in range(len(cluster_result))]

def pickRandomInstances(train_labels, driver_data_dir, cur_driver):
	pos_cnt = len([l for l in train_labels if l == 1])
	neg_cnt = len([l for l in train_labels if l == 0])
	other_drivers = [f for f in os.listdir(driver_data_dir) if f != cur_driver]
	random_data_matrix = []
	while neg_cnt < pos_cnt*2:
		random_driver = other_drivers[random.randint(0, len(other_drivers)-1)]
		all_trips = os.listdir(os.path.join(driver_data_dir, random_driver))
		random_trip = all_trips[random.randint(0, len(all_trips)-1)]
		appendTripFeatureVector(os.path.join(driver_data_dir, random_driver), random_trip, random_data_matrix)
		neg_cnt += 1
	return random_data_matrix

writer = open(result, 'w')
writer.write('driver_trip,prob\n')

process_index = 1
for driver_id in os.listdir(driver_dir):
	if start_index != None and end_index != None:
		if start_index == end_index:
			if str(start_index) != driver_id:
				continue
		else:
			if process_index < start_index:
				process_index += 1
				continue
			if process_index > end_index:
				print str(process_index) + " > " + str(end_index)
				break
		
	trip_list, driver_matrix = extractDriverFeatures(os.path.join(driver_dir, driver_id))
	distMatrix = scipy.cluster.hierarchy.linkage(standardizeCols(driver_matrix), method='centroid', metric='euclidean')
	cluster_result = scipy.cluster.hierarchy.fcluster(distMatrix, threshold*max([i[2] for i in distMatrix]), criterion='distance')
	del(distMatrix)

	train_labels = makeLabelWithClst(cluster_result, len(driver_matrix)*max_pcnt_outlier_per_clst, len(driver_matrix)*max_pcnt_outlier_per_driver)
	random_data_matrix = pickRandomInstances(train_labels, driver_dir, driver_id)

	print "training model..."
	model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=5)
	#model = sklearn.linear_model.LogisticRegression()
	model.fit(driver_matrix + random_data_matrix, train_labels + [0]*len(random_data_matrix))
	
	for i in xrange(len(trip_list)):
		writer.write(driver_id + '_' + trip_list[i] + ',' + str(model.predict_proba(driver_matrix[i])[0][1]) + '\n')
	process_index += 1
writer.close()