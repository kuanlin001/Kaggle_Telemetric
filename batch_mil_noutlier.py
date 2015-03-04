#!/usr/local/bin/python

print 'importing libs...'
import os
import random
import sys

platform = sys.platform

start_index = None
end_index = None

if(platform.upper().startswith('WIN')):
	work_dir = 'C:\\Users\\kuanlin\\Desktop\\Kaggle\\contest_data\\clustering_with_boosting\\'
else:
	work_dir = '/projects/dataconductor_com/broadcom/bin/playground/tele/'

output_folder = os.path.join(work_dir, 'batch_noutlier')
this_script = os.path.join(work_dir, 'batch_mil_noutlier.py')
feature_file = os.path.join(work_dir, 'feature_v2.csv')
filename_token = ""
last_driver = 3612
batch_size = 10

def mergeResults(filepath):
	writer = open(filepath, 'w')
	writer.write('driver_trip,prob\n')
	for filename in os.listdir(output_folder):
		if not filename.upper().endswith('.CSV'):
			continue
		for line in open(os.path.join(output_folder, filename), 'r'):
			line = line.strip()
			if line == '' or line.lower().startswith('driver_trip'):
				continue
			writer.write(line + '\n')
	writer.close()

if len(sys.argv) == 3 and sys.argv[1].lower().strip() == '-run':
	start_index = int(sys.argv[2].split('-')[0])
	end_index = int(sys.argv[2].split('-')[1])
	print "start at index: " + str(start_index)
	print "end at index: " + str(end_index)
elif len(sys.argv) == 4 and sys.argv[1].lower().strip() == '-run':
	start_index = int(sys.argv[2].split('-')[0])
	end_index = int(sys.argv[2].split('-')[1])
	filename_token = sys.argv[3]
	print "start at index: " + str(start_index)
	print "end at index: " + str(end_index)
elif len(sys.argv) == 3 and sys.argv[1].lower().strip() == '-rundist':
	if(platform.upper().startswith('WIN')):
		print "distributed processing only supported on Linux, exiting..."
		sys.exit(1)
	num_jobs = int(sys.argv[2])
	batch_size = int(last_driver/num_jobs)
	start_point = 0
	end_point = 0
	while True:
		start_point = end_point + 1
		end_point = start_point + batch_size
		cmd = "bsub -R rhel50 " + this_script + " -run " + str(start_point) + "-" + str(end_point)
		print "running command: " + cmd
		os.system(cmd)
		if end_point >= last_driver:
			break
	print "remote run job submitted"
	sys.exit(0)
elif len(sys.argv) == 2 and sys.argv[1].lower().strip() == '-mergeresults':
	mergeResults(os.path.join(work_dir, "merged_result.csv"))
	print "result merged to: " + os.path.join(work_dir, "merged_result.csv")
	sys.exit(0)
elif len(sys.argv) == 4 and sys.argv[1].lower().strip() == '-testlocal':
	driver_id = sys.argv[2].strip()
	cmd = this_script + " -run " + driver_id + "-" + driver_id + " " + sys.argv[3]
	print "running command: " + cmd
	os.system(cmd)
	print "test run complete"
	sys.exit(0)
elif len(sys.argv) > 1:
	print "unrecognised arguments, exiting..."
	sys.exit(1)

result = os.path.join(output_folder, str(start_index) + '-' + str(end_index) + '_run_result' + filename_token + '.csv')

def loadFeatures():
	driver_trip_features = {}
	for line in open(feature_file, 'r'):
		line = line.strip()
		if line == '' or line.startswith('driver_trip'):
			continue
		lineArr = line.split(',')
		driver_id = lineArr[0].split('_')[0]
		trip_id = lineArr[0].split('_')[1]
		if driver_id not in driver_trip_features:
			trip_data = {}
			driver_trip_features[driver_id] = trip_data
		else:
			trip_data = driver_trip_features[driver_id]
		
		dataArr = lineArr[1:]
		#print str(dataArr)
		for i, item in enumerate(dataArr):
			dataArr[i] = float(item)
			
		trip_data[trip_id] = dataArr
	return driver_trip_features
	
def formRandomBatches(input_list, b_size=None):
	if b_size == None:
		b_size = batch_size
	input_index = [i for i in xrange(len(input_list))]
	output_list = []
	while len(input_index) > 0:
		if b_size >= len(input_index):
			sampled = [i for i in input_index]
		else:
			sampled = random.sample(input_index, b_size)
		output_list.append([input_list[i] for i in sampled])
		for i in sampled:
			input_index.remove(i)
	return output_list
	
def getRandomDriverData(dataCnt, driver_trip_features, curr_driver_id):
	random_data_matrix = []
	other_drivers = [d for d in driver_trip_features if d != curr_driver_id]
	neg_cnt = 0
	while neg_cnt < dataCnt:
		random_driver = other_drivers[random.randint(0, len(other_drivers)-1)]
		other_trips = driver_trip_features[random_driver]
		random_trip = other_trips.keys()[random.randint(0, len(other_trips)-1)]
		random_data_matrix.append(other_trips[random_trip])
		neg_cnt += 1
	return random_data_matrix
	
def getOutlierTripList(driver_trip_features, current_driver, outlier_lowerbound):
	print "searching for outliers..."
	trips = driver_trip_features[current_driver]
	trip_list = trips.keys()
	driver_matrix = []
	for trip_id in trip_list:
		driver_matrix.append(trips[trip_id])
	random_data_matrix = getRandomDriverData(len(trip_list)*4, driver_trip_features, current_driver)
	
	model = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
	#model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=250, max_depth=5)
	model.fit(driver_matrix + random_data_matrix, [1]*len(driver_matrix) + [0]*len(random_data_matrix))
	
	probs = []
	for trip_id in trip_list:
		probs.append(model.predict_proba(trips[trip_id])[0][1])
	thold = numpy.percentile(probs, outlier_lowerbound)
	
	outlierTrips = set()
	for i, prob in enumerate(probs):
		if prob <= thold:
			#print str(prob)
			outlierTrips.add(trip_list[i])
	return outlierTrips
	
def getOutlierTripListBatched(driver_trip_features, driver_id, outlier_lowerbound):
	print "searching for outliers with batch algorithm..."
	trips = driver_trip_features[driver_id]
	trip_list = trips.keys()
	batched_trip_list = formRandomBatches(trip_list, b_size=20)
	probs = []
	for batch_index, current_batch_trip in enumerate(batched_trip_list):
		driver_matrix = []
		for other_bath_index, other_batches in enumerate(batched_trip_list):
			if other_bath_index == batch_index:
				continue
			for other_trip in other_batches:
				driver_matrix.append(trips[other_trip])
		#print len(driver_matrix)
		random_data_matrix = getRandomDriverData(len(driver_matrix)*5, driver_trip_features, driver_id)
		#model = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
		model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=3)
		model.fit(driver_matrix + random_data_matrix, [1]*len(driver_matrix) + [0]*len(random_data_matrix))
		for trip_id in current_batch_trip:
			probs.append(model.predict_proba(trips[trip_id])[0][1])
	thold = numpy.percentile(probs, outlier_lowerbound)
	outlierTrips = set()
	for i, prob in enumerate(probs):
		if prob <= thold:
			print str(prob)
			outlierTrips.add(trip_list[i])
	return outlierTrips
	
import sklearn.ensemble
import numpy
print "loading features..."
driver_trip_features = loadFeatures()
writer = open(result, 'w')
writer.write('driver_trip,prob\n')

for driver_id in driver_trip_features:
	if start_index != None and end_index != None:
		d_id = int(driver_id)
		if not (d_id >= start_index and d_id <= end_index):
			continue
		
	print "processing dirver_id " + str(driver_id)
	trips = driver_trip_features[driver_id]
	trip_list = trips.keys()
	#outliers = getOutlierTripListBatched(driver_trip_features, driver_id, 10)
	outliers = getOutlierTripList(driver_trip_features, driver_id, 10)
	print str(outliers)
	batched_trip_list = formRandomBatches(trip_list)
	for batch_index, current_batch_trip in enumerate(batched_trip_list):
		#print "current batch trips: " + str(current_batch_trip)
		driver_matrix = []
		for other_bath_index, other_batches in enumerate(batched_trip_list):
			if other_bath_index == batch_index:
				continue
			for other_trip in other_batches:
				if other_trip not in outliers:
					driver_matrix.append(trips[other_trip])
		
		random_data_matrix = getRandomDriverData(len(driver_matrix)*2, driver_trip_features, driver_id)
		#print "training model..."
		#model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
		model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=250, max_depth=5)
		
		model.fit(driver_matrix + random_data_matrix, [1]*len(driver_matrix) + [0]*len(random_data_matrix))
		for trip_id in current_batch_trip:
			writer.write(driver_id + '_' + trip_id + ',' + str(model.predict_proba(trips[trip_id])[0][1]) + '\n')
writer.close()