import os, time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import ifilter

from arch import Architecture
from dice import DiceLoss, DiceCoefficient

class Model(object):

    def __init__(self, labels, load_model_path='', usegpu=True):

        self.labels = labels
        self.n_classes = len(labels)
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

    def __define_criterion(self, class_weights, criterion='CE'):
        assert criterion in ['CE', 'Dice', 'Multi', None]

        smooth = 1.0

        if type(criterion) == type(None):
            self.criterion_dice_coeff = DiceCoefficient(smooth=smooth)

            if self.usegpu:
                self.criterion_dice_coeff = self.criterion_dice_coeff.cuda()
                return

        if type(class_weights) != type(None):
            class_weights = self.__define_variable(torch.FloatTensor(class_weights))
            if criterion == 'CE':
                self.criterion_ce = torch.nn.CrossEntropyLoss(class_weights)
            elif criterion == 'Dice':
                self.criterion_dice = DiceLoss(weight=class_weights, smooth=smooth)
            elif criterion == 'Multi':
                self.criterion_ce = torch.nn.CrossEntropyLoss(class_weights)
                self.criterion_dice = DiceLoss(weight=class_weights, smooth=smooth)
        else:
            if criterion == 'CE':
                self.criterion_ce = torch.nn.CrossEntropyLoss()
            elif criterion == 'Dice':
                self.criterion_dice = DiceLoss(smooth=smooth)
            elif criterion == 'Multi':
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_dice = DiceLoss(smooth=smooth)

        self.criterion_dice_coeff = DiceCoefficient(smooth=smooth)

        if self.usegpu:
            if criterion == 'CE':
                self.criterion_ce = self.criterion_ce.cuda()
            elif criterion == 'Dice':
                self.criterion_dice = self.criterion_dice.cuda()
            elif criterion == 'Multi':
                self.criterion_ce = self.criterion_ce.cuda()
                self.criterion_dice = self.criterion_dice.cuda()

            self.criterion_dice_coeff = self.criterion_dice_coeff.cuda()

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

    def __minibatch(self, train_test_iter, clip_grad_norm, criterion_type, train_cnn=True, mode='training'):
        assert mode in ['training', 'test'], 'Mode must be either "training" or "test"'

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

        if mode == 'training':
            if criterion_type == 'CE':
                _, gpu_annotations_criterion_ce = gpu_annotations.max(3)
                cost = self.criterion_ce(predictions.permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes),
                                         gpu_annotations_criterion_ce.view(-1))
            elif criterion_type == 'Dice':
                gpu_annotations_criterion_dice = gpu_annotations.permute(0, 3, 1, 2).contiguous()
                cost = self.criterion_dice(predictions, gpu_annotations_criterion_dice)
            elif criterion_type == 'Multi':
                _, gpu_annotations_criterion_ce = gpu_annotations.max(3)
                cost_ce = self.criterion_ce(predictions.permute(0, 2, 3, 1).contiguous().view(-1, self.n_classes),
                                            gpu_annotations_criterion_ce.view(-1))
                gpu_annotations_criterion_dice = gpu_annotations.permute(0, 3, 1, 2).contiguous()
                cost_dice = self.criterion_dice(predictions, gpu_annotations_criterion_dice)
                cost = cost_ce + cost_dice
        else:
            gpu_annotations_criterion_dice = gpu_annotations.permute(0, 3, 1, 2).contiguous()
            cost = self.criterion_dice_coeff(predictions, gpu_annotations_criterion_dice)

        if mode == 'training':
            self.model.zero_grad()
            cost.backward()
            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), clip_grad_norm)
            self.optimizer.step()

        return cost, predictions, cpu_annotations

    def __test(self, test_loader):

        print '***** Testing *****'

        n_minibatches = len(test_loader)

        test_iter = iter(test_loader)
        n_correct, n_total = 0.0, 0.0
        dice_coefficients = []

        for minibatch_index in range(n_minibatches):
            dice_coeff, predictions, cpu_annotations = self.__minibatch(test_iter, 0.0, None, False, mode='test')

            _, predictions = predictions.max(1)
            _, cpu_annotations = cpu_annotations.max(3)
            dice_coeff = dice_coeff.data

            n_correct += torch.sum(predictions.data.cpu() == cpu_annotations)
            n_total += predictions.numel()

            dice_coefficients.extend(dice_coeff)

        dice_coefficients = torch.stack(dice_coefficients, dim=0).mean(dim=0)

        accuracy = n_correct / n_total

        print 'Test Accuracy: {}'.format(accuracy)
        print 'Dice Coefficients:'
        for i, coeff in enumerate(dice_coefficients):
            label_name = self.labels[np.where(self.labels[:, 0].astype('int') == i)[0][0]][1]
            print '* {} : {}'.format(label_name, coeff)

        mean_dice_coeff = dice_coefficients[1:].mean() # Discard bg class when calculating mean

        return accuracy, mean_dice_coeff

    def fit(self, criterion_type, learning_rate, weight_decay, clip_grad_norm, lr_drop_factor, lr_drop_patience, optimizer,
            train_cnn, n_epochs, class_weights, train_loader, test_loader, model_save_path):

        training_log_file = open(os.path.join(model_save_path, 'training.log'), 'w')
        validation_log_file = open(os.path.join(model_save_path, 'validation.log'), 'w')

        training_log_file.write('Epoch,Loss,Accuracy\n')
        validation_log_file.write('Epoch,DiceCoefficient,Accuracy\n')

        train_loss_averager = Model.__get_loss_averager()

        self.__define_criterion(class_weights, criterion=criterion_type)
        self.__define_optimizer(learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, optimizer=optimizer)

        self.__test(test_loader)

        best_val_dice_coeff, best_val_acc = 0.0, 0.0
        for epoch in range(n_epochs):
            epoch_start = time.time()

            train_iter = iter(train_loader)
            n_minibatches = len(train_loader)

            minibatch_index = 0
            train_n_correct, train_n_total = 0.0, 0.0
            while minibatch_index < n_minibatches:
                minibatch_cost, minibatch_predictions, minibatch_cpu_annotations = self.__minibatch(train_iter, clip_grad_norm, criterion_type,
                                                                                                    train_cnn=train_cnn, mode='training')

                _, minibatch_predictions = minibatch_predictions.max(1)
                _, minibatch_cpu_annotations = minibatch_cpu_annotations.max(3)
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

            val_accuracy, val_dice_coeff = self.__test(test_loader)

            self.lr_scheduler.step(val_dice_coeff)

            is_best_model_dice_coeff = val_dice_coeff >= best_val_dice_coeff
            is_best_model_acc = val_accuracy >= best_val_acc
            is_best_model = is_best_model_dice_coeff or is_best_model_acc

            if is_best_model:
                if is_best_model_dice_coeff:
                    best_val_dice_coeff = val_dice_coeff
                if is_best_model_acc:
                    best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), os.path.join(model_save_path, 'model_{}_{}_{}.pth'.format(epoch, val_dice_coeff, val_accuracy)))

            training_log_file.write('{},{},{}\n'.format(epoch, train_loss, train_accuracy))
            validation_log_file.write('{},{},{}\n'.format(epoch, val_dice_coeff, val_accuracy))
            training_log_file.flush()
            validation_log_file.flush()

            train_loss_averager.reset()
            train_n_correct, train_n_total = 0.0, 0.0

        training_log_file.close()
        validation_log_file.close()

    def test(self, class_weights, test_loader):

        self.__define_criterion(class_weights, criterion=None)
        test_accuracy, test_dice_coeff = self.__test(test_loader)

        return test_accuracy, test_dice_coeff

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

