import os, time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import ifilter

from arch import Architecture

class Model(object):

    def __init__(self, n_classes, load_model_path='', usegpu=True):

        self.n_classes = n_classes
        self.load_model_path = load_model_path
        self.usegpu = usegpu

        self.model = Architecture(self.n_classes, usegpu=self.usegpu)

        self.__load_weights()

        if self.usegpu:
            cudnn.benchmark = True
            self.model.cuda()
            #self.model = torch.nn.DataParallel(self.model, device_ids=range(self.ngpus))

        print self.model

    def __load_weights(self):

        #def weights_initializer(m):
        #    """Custom weights initialization"""
        #    classname = m.__class__.__name__
        #    if classname.find('Linear') != -1:
        #        m.weight.data.normal_(0.0, 0.001)
        #        m.bias.data.zero_()

        if self.load_model_path != '':
            assert os.path.isfile(self.load_model_path), 'Model : {} does not exists!'.format(self.load_model_path)
            print 'Loading model from {}'.format(self.load_model_path)

            """model_state_dict = self.model.state_dict()

            pretrained_state_dict = torch.load(self.load_model_path)
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if (k in model_state_dict) and (not k.startswith('linear'))}

            model_state_dict.update(pretrained_state_dict)
            self.model.load_state_dict(model_state_dict)"""

            self.model.load_state_dict(torch.load(self.load_model_path))

            """if self.usegpu:
                loaded_model = torch.load(self.load_model_path)
            else:
                loaded_model = torch.load(self.load_model_path, map_location=lambda storage, loc: storage)
            loaded_model_layer_keys = loaded_model.keys()
            for layer_key in loaded_model_layer_keys:
                if layer_key.startswith('module.'):
                    new_layer_key = '.'.join(layer_key.split('.')[1:])
                    loaded_model[new_layer_key] = loaded_model.pop(layer_key)
            self.model.load_state_dict(loaded_model)"""
        #else:
        #    self.model.apply(weights_initializer)

    def __define_variable(self, tensor, volatile=False):
        return Variable(tensor, volatile=volatile)

    def __define_input_variables(self, features, labels, volatile=False):
        features_var = self.__define_variable(features, volatile=volatile)
        labels_var = self.__define_variable(labels, volatile=volatile)

        return features_var, labels_var

    def __define_criterion(self, class_weights):
        if type(class_weights) != type(None):
            class_weights = torch.FloatTensor(class_weights)
            self.criterion = torch.nn.CrossEntropyLoss(class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if self.usegpu:
            self.criterion = self.criterion.cuda()

    def __define_optimizer(self, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, optimizer='Adam'):
        assert optimizer in ['RMSprop', 'Adam', 'Adadelta', 'SGD']

        parameters = ifilter(lambda p: p.requires_grad, self.model.parameters())

        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_drop_factor, patience=lr_drop_patience, verbose=True)

    @staticmethod
    def __get_loss_averager():
        return averager()

    def __minibatch(self, train_test_iter, clip_grad_norm, train_cnn=True, mode='training'):
        assert mode in ['training', 'validation'], 'Mode must be either "training" or "validation"'

        if mode == 'training':
            for param in self.model.parameters():
                param.requires_grad = True
            if not train_cnn:
                for param in self.model.cnn.parameters():
                    param.requires_grad = False
            self.model.train()
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        cpu_images, cpu_annotations = train_test_iter.next()
        cpu_images = cpu_images.contiguous()
        cpu_annotations = cpu_annotations.contiguous()

        if self.usegpu:
            gpu_images = cpu_images.cuda(async=True)
            gpu_annotations = cpu_annotations.cuda(async=True)
        else:
            gpu_images = cpu_images
            gpu_annotations = cpu_annotations

        if mode == 'training':
            gpu_images, gpu_annotations = self.__define_input_variables(gpu_images, gpu_annotations)
        else:
            gpu_images, gpu_annotations = self.__define_input_variables(gpu_images, gpu_annotations, volatile=True)

        predictions = self.model(gpu_images)

        cost = self.criterion(predictions.permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes), gpu_annotations.view(-1))

        if mode == 'training':
            self.model.zero_grad()
            cost.backward()
            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), clip_grad_norm)
            self.optimizer.step()

        return cost, predictions, cpu_annotations

    def __validate(self, test_loader):
        n_minibatches = len(test_loader)

        test_loss_averager = Model.__get_loss_averager()

        test_iter = iter(test_loader)
        n_correct, n_total = 0.0, 0.0

        for minibatch_index in range(n_minibatches):
            cost, predictions, cpu_annotations = self.__minibatch(test_iter, 0.0, False, mode='validation')
            test_loss_averager.add(cost)

            _, predictions = predictions.max(1)
            n_correct += torch.sum(predictions.data.cpu() == cpu_annotations)
            n_total += predictions.numel()

        loss = test_loss_averager.val()
        accuracy = n_correct / n_total

        print 'Validation Loss: {}, Accuracy: {}'.format(loss, accuracy)

        return accuracy, loss

    def fit(self, learning_rate, weight_decay, clip_grad_norm, lr_drop_factor, lr_drop_patience, optimizer,
            train_cnn, n_epochs, class_weights, train_loader, test_loader, model_save_path):

        training_log_file = open(os.path.join(model_save_path, 'training.log'), 'w')
        validation_log_file = open(os.path.join(model_save_path, 'validation.log'), 'w')

        training_log_file.write('Epoch,Loss,Accuracy\n')
        validation_log_file.write('Epoch,Loss,Accuracy\n')

        train_loss_averager = Model.__get_loss_averager()

        self.__define_criterion(class_weights)
        self.__define_optimizer(learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, optimizer=optimizer)

        self.__validate(test_loader)

        best_val_loss, best_val_acc = np.Inf, 0.0
        for epoch in range(n_epochs):
            epoch_start = time.time()

            train_iter = iter(train_loader)
            n_minibatches = len(train_loader)

            minibatch_index = 0
            train_n_correct, train_n_total = 0.0, 0.0
            while minibatch_index < n_minibatches:
                minibatch_cost, minibatch_predictions, minibatch_cpu_annotations = self.__minibatch(train_iter, clip_grad_norm, 
                                                                                                    train_cnn=train_cnn, mode='training')

                _, minibatch_predictions = minibatch_predictions.max(1)
                train_n_correct += torch.sum(minibatch_predictions.data.cpu() == minibatch_cpu_annotations)
                train_n_total += minibatch_predictions.numel()

                train_loss_averager.add(minibatch_cost)
                minibatch_index += 1

            train_accuracy = train_n_correct / train_n_total
            train_loss = train_loss_averager.val()

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start

            print '[{}] [{}/{}] Loss : {} - Accuracy : {}'.format(epoch_duration, epoch, n_epochs, train_loss,
                                                                  train_accuracy)

            val_accuracy, val_loss = self.__validate(test_loader)

            self.lr_scheduler.step(val_accuracy)

            is_best_model_loss = val_loss <= best_val_loss
            is_best_model_acc = val_accuracy >= best_val_acc
            is_best_model = is_best_model_loss or is_best_model_acc

            if is_best_model:
                if is_best_model_loss:
                    best_val_loss = val_loss
                if is_best_model_acc:
                    best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), os.path.join(model_save_path, 'model_{}_{}_{}.pth'.format(epoch, val_loss, val_accuracy)))

            training_log_file.write('{},{},{}\n'.format(epoch, train_loss, train_accuracy))
            validation_log_file.write('{},{},{}\n'.format(epoch, val_loss, val_accuracy))
            training_log_file.flush()
            validation_log_file.flush()

            train_loss_averager.reset()
            train_n_correct, train_n_total = 0.0, 0.0

        training_log_file.close()
        validation_log_file.close()

    def predict(self, images):

        assert len(images.size()) == 4 #b, c, h, w

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        images = images.contiguous()
        if self.usegpu:
            images = images.cuda(async=True)

        images = self.__define_variable(images, volatile=True)

        predictions = self.model(images)

        return predictions

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`."""

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

