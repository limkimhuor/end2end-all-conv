import sys
import numpy as np
from keras.callbacks import Callback
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.optimizers import (
    SGD, RMSprop, Adagrad, Adadelta,
    Adam, Adamax, Nadam
)
from keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping, 
    ModelCheckpoint
)
import keras.backend as K
data_format = K.image_data_format()
from sklearn.metrics import roc_auc_score
from dm_resnet import ResNetBuilder
from dm_multi_gpu import make_parallel


def load_dat_ram(generator, nb_samples):
    samples_seen = 0
    X_list = []
    y_list = []
    w_list = []
    while samples_seen < nb_samples:
        blob_ = generator.next()
        try:
            X,y,w = blob_
            w_list.append(w)
        except ValueError:
            X,y = blob_
        X_list.append(X)
        y_list.append(y)
        samples_seen += len(y)
    try:
        data_set = (np.concatenate(X_list), 
                    np.concatenate(y_list),
                    np.concatenate(w_list))
    except ValueError:
        data_set = (np.concatenate(X_list), 
                    np.concatenate(y_list))

    if len(data_set[0]) != nb_samples:
        raise Exception('Load data into RAM error')

    return data_set


def get_dl_model(net, nb_class=3, use_pretrained=True, resume_from=None, 
                 top_layer_nb=None, weight_decay=.01,
                 bias_multiplier=.1, hidden_dropout=.0, **kw_args):
    '''Load existing DL model or create it from new
    Args:
        kw_args: keyword arguments for creating resnet.
    '''
    if net == 'resnet50':
        from keras.applications.resnet50 import ResNet50 as NNet, preprocess_input
        top_layer_nb = 162 if top_layer_nb is None else top_layer_nb
    elif net == 'vgg16':
        from keras.applications.vgg16 import VGG16 as NNet, preprocess_input
        top_layer_nb = 15 if top_layer_nb is None else top_layer_nb
    elif net == 'vgg19':
        from keras.applications.vgg19 import VGG19 as NNet, preprocess_input
        top_layer_nb = 17 if top_layer_nb is None else top_layer_nb
    elif net == 'xception':
        from keras.applications.xception import Xception as NNet, preprocess_input
        top_layer_nb = 126 if top_layer_nb is None else top_layer_nb
    elif net == 'inception':
        from keras.applications.inception_v3 import InceptionV3 as NNet, preprocess_input
        top_layer_nb = 194 if top_layer_nb is None else top_layer_nb
    else:
        raise Exception("Requested model is not available: " + net)
    weights = 'imagenet' if use_pretrained else None

    if resume_from is not None:
        print "Loading existing model state.",
        sys.stdout.flush()
        model = load_model(resume_from)
        print "Done."
    else:
        print "Loading %s," % (net),
        sys.stdout.flush()
        base_model = NNet(weights=weights, include_top=False, 
                          input_shape=None, pooling='avg')
        x = base_model.output
        if hidden_dropout > 0.:
            x = Dropout(hidden_dropout)(x)
        preds = Dense(nb_class, activation='softmax', 
                      kernel_regularizer=l2(weight_decay), 
                      bias_regularizer=l2(weight_decay*bias_multiplier))(x)
        model = Model(input=base_model.input, output=preds)
        print "Done."

    return model, preprocess_input, top_layer_nb


def create_optimizer(optim_name, lr):
    if optim_name == 'sgd':
        return SGD(lr, momentum=.9, nesterov=True)
    elif optim_name == 'rmsprop':
        return RMSprop(lr)
    elif optim_name == 'adagrad':
        return Adagrad(lr)
    elif optim_name == 'adadelta':
        return Adadelta(lr)
    elif optim_name == 'adamax':
        return Adamax(lr)
    elif optim_name == 'adam':
        return Adam(lr)
    elif optim_name == 'nadam':
        return Nadam(lr)
    else:
        raise Exception('Unknown optimizer name: ' + optim_name)


def do_3stage_training(model, org_model, train_generator, validation_set, 
                       validation_steps, best_model_out, steps_per_epoch, 
                       top_layer_nb=None, net=None,
                       nb_epoch=10, top_layer_epochs=0, all_layer_epochs=0,
                       use_pretrained=True, optim='sgd', init_lr=.01, 
                       top_layer_multiplier=.01, all_layer_multiplier=.0001,
                       es_patience=5, lr_patience=2, auto_batch_balance=True, 
                       nb_class=3,
                       pos_cls_weight=1., neg_cls_weight=1., nb_worker=1,
                       weight_decay2=.01, bias_multiplier=.1, hidden_dropout2=.0):
    '''3-stage DL model training
    '''
    # Create callbacks and class weight.
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, 
                                   verbose=1)
    # best_model += ".{epoch:03d}-{val_acc:.2f}.h5"
    checkpointer = ModelCheckpoint(best_model_out, monitor='val_acc', verbose=1, 
                                   save_best_only=True)
    stdout_flush = DMFlush()
    callbacks = [early_stopping, checkpointer, stdout_flush]
    if optim == 'sgd':
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                      patience=lr_patience, verbose=1)
        callbacks.append(reduce_lr)
    if auto_batch_balance:
        class_weight = None
    elif nb_class == 2:
        class_weight = { 0:1.0, 1:pos_cls_weight }
    elif nb_class == 3:
        class_weight = { 0:1.0, 1:pos_cls_weight, 2:neg_cls_weight }
    else:
        class_weight = None
    if nb_worker == 1:
        pickle_safe = False
    else:
        pickle_safe = True

    # Stage 1: train only the last dense layer if using pretrained model.
    print "Start model training",
    if use_pretrained:
        print "on the last dense layer only"
        for layer in org_model.layers[:-1]:
            layer.trainable = False
    else:
        print "on all layers"
    sys.stdout.flush()
    model.compile(optimizer=create_optimizer(optim, init_lr), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit_generator(
        train_generator, 
        steps_per_epoch=steps_per_epoch, 
        epochs=nb_epoch,
        class_weight=class_weight,
        validation_data=validation_set,
        validation_steps=validation_steps,
        callbacks=callbacks, 
        nb_worker=nb_worker, 
        pickle_safe=pickle_safe,
        verbose=2)
    print "Done."
    try:
        loss_history = hist.history['val_loss']
        acc_history = hist.history['val_acc']
    except KeyError:
        loss_history = []
        acc_history = []
    
    # Stage 2: train only the top layers.
    if use_pretrained:
        print "top layer nb =", top_layer_nb
        for layer in org_model.layers[top_layer_nb:]:
            layer.trainable = True
        # # adjust weight decay and dropout rate for those BN heavy models.
        # if net == 'xception' or net == 'inception' or net == 'resnet50':
        dense_layer = org_model.layers[-1]
        dropout_layer = org_model.layers[-2]
        dense_layer.kernel_regularizer.l2 = weight_decay2
        dense_layer.bias_regularizer.l2 = weight_decay2*bias_multiplier
        dropout_layer.rate = hidden_dropout2
        model.compile(optimizer=create_optimizer(optim, init_lr*top_layer_multiplier), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print "Start training on the top layers only"; sys.stdout.flush()
        hist = model.fit_generator(
            train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=top_layer_epochs,
            class_weight=class_weight,
            validation_data=validation_set,
            validation_steps=validation_steps,
            callbacks=callbacks, 
            nb_worker=nb_worker, 
            pickle_safe=pickle_safe,
            verbose=2, initial_epoch=len(loss_history))
        print "Done."
        try:
            loss_history = np.append(loss_history, hist.history['val_loss'])
            acc_history = np.append(acc_history, hist.history['val_acc'])
        except KeyError:
            pass

    # Stage 3: train all layers.
        for layer in org_model.layers[:top_layer_nb]:
            layer.trainable = True
        model.compile(optimizer=create_optimizer(optim, init_lr*all_layer_multiplier), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print "Start training on all layers"; sys.stdout.flush()
        hist = model.fit_generator(
            train_generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=all_layer_epochs,
            class_weight=class_weight,
            validation_data=validation_set,
            validation_steps=validation_steps,
            callbacks=callbacks, 
            nb_worker=nb_worker, 
            pickle_safe=pickle_safe,
            verbose=2, initial_epoch=len(loss_history))
        print "Done."
        try:
            loss_history = np.append(loss_history, hist.history['val_loss'])
            acc_history = np.append(acc_history, hist.history['val_acc'])
        except KeyError:
            pass
    return model, loss_history, acc_history


def do_2stage_training(model, org_model, train_generator, validation_set, 
                       validation_steps, best_model_out, steps_per_epoch, 
                       top_layer_nb, nb_epoch=10, all_layer_epochs=0,
                       optim='sgd', init_lr=.01, all_layer_multiplier=.1,
                       es_patience=5, lr_patience=2, auto_batch_balance=True, 
                       nb_class=2,
                       pos_cls_weight=1., neg_cls_weight=1., nb_worker=1,
                       auc_checkpointer=None):
    '''2-stage DL model training (for whole images)
    '''
    # Create callbacks and class weight.
    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience, 
                                   verbose=1)
    if auc_checkpointer is None:
        checkpointer = ModelCheckpoint(
            best_model_out, monitor='val_acc', verbose=1, save_best_only=True)
    else:
        checkpointer = auc_checkpointer
    stdout_flush = DMFlush()
    callbacks = [early_stopping, checkpointer, stdout_flush]
    if optim == 'sgd':
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                      patience=lr_patience, verbose=1)
        callbacks.append(reduce_lr)
    if auto_batch_balance:
        class_weight = None
    elif nb_class == 2:
        class_weight = { 0:1.0, 1:pos_cls_weight }
    elif nb_class == 3:
        class_weight = { 0:1.0, 1:pos_cls_weight, 2:neg_cls_weight }
    else:
        class_weight = None
    if nb_worker == 1:
        pickle_safe = False
    else:
        pickle_safe = True

    # Stage 1: train only the top layers.
    print "Top layer nb =", top_layer_nb
    for layer in org_model.layers[:top_layer_nb]:
        layer.trainable = False
    model.compile(optimizer=create_optimizer(optim, init_lr), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print "Start training on the top layers only"; sys.stdout.flush()
    hist = model.fit_generator(
        train_generator, 
        steps_per_epoch=steps_per_epoch, 
        epochs=nb_epoch,
        class_weight=class_weight,
        validation_data=validation_set,
        validation_steps=validation_steps,
        callbacks=callbacks, 
        nb_worker=nb_worker, 
        pickle_safe=pickle_safe,
        verbose=2)
    print "Done."
    try:
        loss_history = hist.history['val_loss']
        acc_history = hist.history['val_acc']
    except KeyError:
        loss_history = []
        acc_history = []

    # Stage 2: train all layers.
    for layer in org_model.layers[:top_layer_nb]:
        layer.trainable = True
    model.compile(optimizer=create_optimizer(optim, init_lr*all_layer_multiplier), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print "Start training on all layers"; sys.stdout.flush()
    hist = model.fit_generator(
        train_generator, 
        steps_per_epoch=steps_per_epoch, 
        epochs=all_layer_epochs,
        class_weight=class_weight,
        validation_data=validation_set,
        validation_steps=validation_steps,
        callbacks=callbacks, 
        nb_worker=nb_worker, 
        pickle_safe=pickle_safe,
        verbose=2, initial_epoch=len(loss_history))
    print "Done."
    try:
        loss_history = np.append(loss_history, hist.history['val_loss'])
        acc_history = np.append(acc_history, hist.history['val_acc'])
    except KeyError:
        pass

    return model, loss_history, acc_history


class DMMetrics(object):
    '''Classification metrics for the DM challenge
    '''

    @staticmethod
    def sensitivity(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pos = K.round(K.clip(y_true, 0, 1))
        tp = K.sum(y_pos * y_pred_pos)
        pos = K.sum(y_pos)

        return tp / (pos + K.epsilon())

    @staticmethod
    def specificity(y_true, y_pred):
        y_pred_neg = 1 - K.round(K.clip(y_pred, 0, 1))
        y_neg = 1 - K.round(K.clip(y_true, 0, 1))
        tn = K.sum(y_neg * y_pred_neg)
        neg = K.sum(y_neg)

        return tn / (neg + K.epsilon())


class DMAucModelCheckpoint(Callback):
    '''Model checkpointer using AUROC score
    '''

    def __init__(self, filepath, test_data, test_samples=None, 
                 batch_size=None):
        super(DMAucModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.test_data = test_data
        if isinstance(test_data, tuple):
            if batch_size is None:
                raise Exception('batch_size must be specified when ' + \
                                'validation data is loaded into RAM')
        elif test_samples is None:
            raise Exception('test_samples must be specified when ' + \
                            'test_data is a generator')
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.best_epoch = 0
        self.best_auc = -1.
        self.best_all_auc = None

    @staticmethod
    def calc_test_auc(test_set, model, batch_size=None, test_samples=None):
        if isinstance(test_set, tuple):
            if batch_size is None:
                raise Exception('batch_size must be specified when ' + \
                                'test set is loaded into RAM')
            y_true = test_set[1]
            y_pred = model.predict(test_set[0], batch_size)
            if len(test_set) > 2:
                weights = test_set[2]
            else:
                weights = None
        else:
            if test_samples is None:
                raise Exception('test_samples must be specified when ' + \
                                'test set is a generator')
            test_set.reset()
            samples_seen = 0
            y_list = []
            pred_list = []
            wei_list = []
            while samples_seen < test_samples:
                res = next(test_set)
                if len(res) > 2:
                    w = res[2]
                    wei_list.append(w)
                X, y = res[:2]
                samples_seen += len(y)
                y_list.append(y)
                pred_list.append(model.predict_on_batch(X))
            y_true = np.concatenate(y_list)
            y_pred = np.concatenate(pred_list)
            if len(wei_list) > 0:
                weights = np.concatenate(wei_list)
            else:
                weights = None
        # Calculate AUC score.
        # import pdb; pdb.set_trace()
        try:
            auc = roc_auc_score(y_true, y_pred, average=None, 
                                sample_weight=weights)
        except ValueError:
            auc = .0
        return auc

    def on_epoch_end(self, epoch, logs={}):
        auc = self.calc_test_auc(self.test_data, self.model, self.batch_size, 
                                 self.test_samples)
        # Calculate AUC for pos and neg classes on non-background cases.
        # if y_pred.shape[1] == 3:
        #     non_bkg_idx = np.where(y_true[:,0]==0)[0]
        #     sample_weight = None if weights is None else weights[non_bkg_idx]
        #     try:
        #         non_bkg_auc_pos = roc_auc_score(
        #             y_true[non_bkg_idx, 1], y_pred[non_bkg_idx, 1], 
        #             sample_weight=sample_weight)
        #     except ValueError:
        #         non_bkg_auc_pos = .0
        #     try:
        #         non_bkg_auc_neg = roc_auc_score(
        #             y_true[non_bkg_idx, 2], y_pred[non_bkg_idx, 2], 
        #             sample_weight=sample_weight)
        #     except ValueError:
        #         non_bkg_auc_neg = .0
        # import pdb; pdb.set_trace()
        # if isinstance(auc, float):
        #     print " - Epoch:%d, AUROC: %.4f" % (epoch + 1, auc)
        # elif len(auc) == 2:
        #     auc = auc[1]
        #     print " - Epoch:%d, AUROC: %.4f" % (epoch + 1, auc)
        # elif len(auc) == 3:
        #     print " - Epoch:%d, AUROC: bkg - %.4f, pos - %.4f, neg - %.4f" \
        #             % (epoch + 1, auc[0], auc[1], auc[2])
        #     print " - non-bkg pos AUROC: %.4f, neg AUROC: %.4f" \
        #             % (non_bkg_auc_pos, non_bkg_auc_neg)
        # else:
        #     raise Exception("Unknown auc format: " + str(auc))
        epoch_auc = np.mean(auc)
        print " - Epoch:%d, AUROC:%s, mean=%.4f" % (epoch + 1, str(auc), epoch_auc)
        sys.stdout.flush()
        # epoch_auc = non_bkg_auc_pos if y_pred.shape[1] == 3 else auc
        if epoch_auc > self.best_auc:
            self.best_epoch = epoch + 1
            self.best_auc = epoch_auc
            self.best_all_auc = auc
            if self.filepath != "NOSAVE":
                self.model.save(self.filepath)

    def on_train_end(self, logs={}):
        print "\n>>> Found best AUROC: %.4f at epoch: %d, saved to: %s <<<" % \
            (self.best_auc, self.best_epoch, self.filepath)
        print ">>> AUROC for all cls:", str(self.best_all_auc), "<<<"
        sys.stdout.flush()


class DMFlush(Callback):
    '''A callback does nothing but flushes stdout after each epoch
    '''
    def __init__(self):
        super(DMFlush, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()


