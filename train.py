import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import argparse
sys.path.append('./')
from distil.utils.models.resnet import ResNet18
from distil.utils.data_handler import DataHandler_Points, DataHandler_MNIST, DataHandler_CIFAR10, \
										DataHandler_FASHION_MNIST, DataHandler_SVHN, DataHandler_STL10, \
                                        DataHandler_CIFAR100
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, LeastConfidence, \
                                        MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, KMeansSampling, \
                                        BALDDropout, FASS
from distil.utils.models.simple_net import TwoLayerNet
from distil.utils.dataset import get_dataset
from distil.utils.train_helper import data_train
from distil.utils.config_helper import read_config_file
import time
import pickle

class TrainClassifier:
	
	def __init__(self, config_file):
		self.config_file = config_file
		self.config = read_config_file(config_file)

	def getModel(self, model_config):

		if model_config['architecture'] == 'resnet18':

			if ('target_classes' in model_config) and ('channel' in model_config):
				net = ResNet18(num_classes = model_config['target_classes'], channels = model_config['channel'])
			elif 'target_classes' in model_config:
				net = ResNet18(num_classes = model_config['target_classes'])
			else:
				net = ResNet18()
		
		elif model_config['architecture'] == 'two_layer_net':
			net = TwoLayerNet(model_config['input_dim'], model_config['target_classes'], model_config['hidden_units_1'])

		return net

	def libsvm_file_load(self, path,dim, save_data=False):

	    data = []
	    target = []
	    with open(path) as fp:
	       line = fp.readline()
	       while line:
	        temp = [i for i in line.strip().split(" ")]
	        target.append(int(float(temp[0]))) # Class Number. # Not assumed to be in (0, K-1)
	        temp_data = [0]*dim
	        
	        for i in temp[1:]:
	            ind,val = i.split(':')
	            temp_data[int(ind)-1] = float(val)
	        data.append(temp_data)
	        line = fp.readline()
	    X_data = np.array(data,dtype=np.float32)
	    Y_label = np.array(target)
	    if save_data:
	        # Save the numpy files to the folder where they come from
	        data_np_path = path + '.data.npy'
	        target_np_path = path + '.label.npy'
	        np.save(data_np_path, X_data)
	        np.save(target_np_path, Y_label)

	    return (X_data, Y_label)

	def getData(self, data_config):

		# print(data_config)
		if data_config['name'] == 'cifar10':
			data_set_name = 'CIFAR10'
			download_path = './downloaded_data/'
			X, y, X_test, y_test = get_dataset(data_set_name, download_path)
			handler = DataHandler_CIFAR10
			y_test = y_test.numpy()
			y = y.numpy()

		elif data_config['name'] == 'mnist':
			data_set_name = 'MNIST'
			download_path = './downloaded_data/'
			X, y, X_test, y_test = get_dataset(data_set_name, download_path)
			handler = DataHandler_MNIST
			X = X.numpy()
			y = y.numpy()
			X_test = X_test.numpy()
			y_test = y_test.numpy()

		elif data_config['name'] == 'fmnist':
			
			data_set_name = 'FASHION_MNIST'
			download_path = './downloaded_data/'
			X, y, X_test, y_test = get_dataset(data_set_name, download_path)
			handler = DataHandler_FASHION_MNIST
			y_test = y_test.numpy()
			y = y.numpy()

		elif data_config['name'] == 'svhn':
			
			data_set_name = 'SVHN'
			download_path = './downloaded_data/'
			X, y, X_test, y_test = get_dataset(data_set_name, download_path)
			handler = DataHandler_SVHN
			y_test = y_test.numpy()
			y = y.numpy()

		elif data_config['name'] == 'cifar100':
			
			data_set_name = 'CIFAR100'
			download_path = './downloaded_data/'
			X, y, X_test, y_test = get_dataset(data_set_name, download_path)
			handler = DataHandler_CIFAR100
			y_test = y_test.numpy()
			y = y.numpy()

		elif data_config['name'] == 'stl10':
			
			data_set_name = 'STL10'
			download_path = './downloaded_data/'
			X, y, X_test, y_test = get_dataset(data_set_name, download_path)
			handler = DataHandler_STL10
			y_test = y_test.numpy()
			y = y.numpy()

		elif data_config['name'] == 'satimage':

		    trn_file = '../datasets/satimage/satimage.scale.trn'
		    val_file = '../datasets/satimage/satimage.scale.val'
		    tst_file = '../datasets/satimage/satimage.scale.tst'
		    data_dims = 36
		    num_cls = 6

		    X, y = self.libsvm_file_load(trn_file, dim=data_dims)
		    X_test, y_test = self.libsvm_file_load(tst_file, dim=data_dims)

		    y -= 1  # First Class should be zero
		    y_test -= 1  # First Class should be zero

		    sc = StandardScaler()
		    X = sc.fit_transform(X)
		    X_test = sc.transform(X_test)
		    handler = DataHandler_Points

		elif data_config['name'] == 'ijcnn1':
			
			trn_file = '../datasets/ijcnn1/ijcnn1.trn'
			val_file = '../datasets/ijcnn1/ijcnn1.val'
			tst_file = '../datasets/ijcnn1/ijcnn1.tst'
			data_dims = 22
			num_cls = 2
			
			X, y = self.libsvm_file_load(trn_file, dim=data_dims)
			X_test, y_test = self.libsvm_file_load(tst_file, dim=data_dims) 

			# The class labels are (-1,1). Make them to (0,1)
			y[y < 0] = 0
			y_test[y_test < 0] = 0    

			sc = StandardScaler()
			X = sc.fit_transform(X)
			X_test = sc.transform(X_test)
			handler = DataHandler_Points

		return X, y, X_test, y_test, handler

	def write_logs(self, logs, save_location, rd):
	  
	  file_path = save_location
	  with open(file_path, 'a') as f:
	    f.write('---------------------\n')
	    f.write('Round '+str(rd)+'\n')
	    f.write('---------------------\n')
	    for key, val in logs.items():
	      if key == 'Training':
	        f.write(str(key)+ '\n')
	        for epoch in val:
	          f.write(str(epoch)+'\n')       
	      else:
	        f.write(str(key) + ' - '+ str(val) +'\n')

	def train_classifier(self):
		
		net = self.getModel(self.config['model'])
		X, y, X_test, y_test, handler = self.getData(self.config['dataset'])
		selected_strat = self.config['active_learning']['strategy']
		budget = self.config['active_learning']['budget']
		start = self.config['active_learning']['initial_points']
		n_rounds = self.config['active_learning']['rounds']
		nclasses = self.config['model']['target_classes']
		strategy_args = self.config['active_learning']['strategy_args'] 
	    
		nSamps = np.shape(X)[0]
		np.random.seed(42)
		start_idxs = np.random.choice(nSamps, size=start, replace=False)
		X_tr = X[start_idxs]
		X_unlabeled = np.delete(X, start_idxs, axis = 0)
		y_tr = y[start_idxs]
		y_unlabeled = np.delete(y, start_idxs, axis = 0)
		
		if 'islogs' in self.config['train_parameters']:
			islogs = self.config['train_parameters']['islogs']
			save_location = self.config['train_parameters']['logs_location']
		else:
			islogs = False

		dt = data_train(X_tr, y_tr, net, handler, self.config['train_parameters'])

		if selected_strat == 'badge':
		    strategy = BADGE(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'glister':
		    strategy = GLISTER(X_tr, y_tr, X_unlabeled, net, handler,nclasses, strategy_args,valid=False,\
		                typeOf='Diversity',lam=10)
		elif selected_strat == 'entropy_sampling':
		    strategy = EntropySampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses)
		elif selected_strat == 'margin_sampling':
		    strategy = MarginSampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses)
		elif selected_strat == 'least_confidence':
		    strategy = LeastConfidence(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'coreset':
		    strategy = CoreSet(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'random_sampling':
		    strategy = RandomSampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'fass':
			strategy = FASS(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'bald_dropout':
		    strategy = BALDDropout(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'adversarial_bim':
		    strategy = AdversarialBIM(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'kmeans_sampling':
		    strategy = KMeansSampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'baseline_sampling':
		    strategy = BaselineSampling(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		elif selected_strat == 'adversarial_deepfool':
		    strategy = AdversarialDeepFool(X_tr, y_tr, X_unlabeled, net, handler, nclasses, strategy_args)
		else:
			raise IOError('Enter Valid Strategy!')

		if islogs:
			clf, train_logs= dt.train()
		else:
			clf = dt.train()
		strategy.update_model(clf)
		y_pred = strategy.predict(X_test).numpy()

		acc = np.zeros(n_rounds)
		acc[0] = round(1.0*(y_test == y_pred).sum().item() / len(y_test), 3)

		if islogs:
			logs = {}
			logs['Training Points'] = X_tr.shape[0]
			logs['Test Accuracy'] =  str(round(acc[0]*100, 2))
			logs['Training'] = train_logs
			self.write_logs(logs, save_location, 0)
		
		print('***************************')
		print('Starting Training..')
		print('***************************')
	    ##User Controlled Loop
		for rd in range(1, n_rounds):
			print('***************************')
			print('Round', rd)
			print('***************************')		
			logs = {}
			t0 = time.time()
			idx = strategy.select(budget)
			t1 = time.time()
			strategy.save_state('./state.pkl')

		    #Adding new points to training set
			X_tr = np.concatenate((X_tr, X_unlabeled[idx]), axis=0)
			X_unlabeled = np.delete(X_unlabeled, idx, axis = 0)

			#Human In Loop, Assuming user adds new labels here
			y_tr = np.concatenate((y_tr, y_unlabeled[idx]), axis = 0)
			y_unlabeled = np.delete(y_unlabeled, idx, axis = 0)

			print('Total training points in this round', X_tr.shape[0])

			#Reload state and start training
			strategy.load_state('./state.pkl')
			strategy.update_data(X_tr, y_tr, X_unlabeled)
			dt.update_data(X_tr, y_tr)

			if islogs:
				clf, train_logs= dt.train()
			else:
				clf = dt.train()
			t2 = time.time()
			strategy.update_model(clf)
			y_pred = strategy.predict(X_test).numpy()
			acc[rd] = round(1.0 * (y_test == y_pred).sum().item() / len(y_test), 3)
			print('Testing Accuracy:', acc[rd])

			if islogs:
			    logs['Training Points'] = X_tr.shape[0]
			    logs['Test Accuracy'] =  str(round(acc[rd]*100, 2))
			    logs['Selection Time'] = str(t1 - t0)
			    logs['Trainining Time'] = str(t2 - t1) 
			    logs['Training'] = train_logs
			    self.write_logs(logs, save_location, rd)

		print('Training Completed!')
		with open('./final_model.pkl', 'wb') as save_file:
			pickle.dump(clf.state_dict(), save_file)
		print('Model Saved!')

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--config_path', required=True, help="Path to the config file")
  args = parser.parse_args()
  tc = TrainClassifier(args.config_path)
  tc.train_classifier()


# tc = TrainClassifier('./configs/config_2lnet_satimage_randomsampling.json')
# # tc = TrainClassifier('./configs/config_cifar10_marginsampling.json')
# tc.train_classifier()