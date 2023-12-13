import copy
import datetime
from sklearn.model_selection import KFold
from train_model import *
from utils import Averager

ROOT = os.getcwd()
_, os.environ['CUDA_VISIBLE_DEVICES'] = config.set_config()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = os.path.join(args.save_path, 'result')
        ensure_path(result_path)
        self.text_file = os.path.join(result_path,
                                      "results_{}.txt".format(args.dataset))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)training_rate:" + str(args.training_rate) +
                   "\n5)pool:" + str(args.pool) +
                   "\n6)num_epochs:" + str(args.max_epoch) +
                   "\n7)batch_size:" + str(args.batch_size) +
                   "\n8)dropout:" + str(args.dropout) +
                   "\n9)hidden_node:" + str(args.hidden) +
                   "\n10)input_shape:" + str(args.input_shape) +
                   "\n11)class:" + str(args.label_type) +
                   "\n12)T:" + str(args.T) +
                   "\n13)graph-type:" + str(args.graph_type) + '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        param sub: which subject's data to load
        return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = os.path.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        param idx_train: index of training data
        param idx_test: index of testing data
        param data: (segments, 1, channel, data)
        param label: (segments,)
        return: data and label
        """
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        # For DEAP we want to do trial-wise 10-fold, so the idx_train/idx_test is for trials.
        # data: (trial, segment, 1, chan, datapoint)
        # To use the normalization function, we should change the dimension from
        # (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
        data_train = np.concatenate(data_train, axis=0)
        label_train = np.concatenate(label_train, axis=0)

        # the testing data do not need to be concatenated, when doing leave-one-trial-out
        if len(data_test.shape) > 4:
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)

        data_train, data_test = self.normalize(train=data_train, test=data_test)

        # Prepare the data format for training the model
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        param train: training data (sample, 1, chan, datapoint)
        param test: testing data (sample, 1, chan, datapoint)
        return: normalized training and testing data.
        """
        # data: sample x 1 x channel x data
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        param data: training data (segment, 1, channel, data)
        param label: (segments,)
        param train_rate: the percentage of trianing data
        param random: bool, whether to shuffle the training data before get the validation data
        return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def n_fold_CV(self, subject=None, fold=4, shuffle=True, reproduce=False):
        """
        this function achieves n-fold cross-validation
        param subject: how many subjects to load
        param fold: how many fold.
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1

        tta_trial = []   # for trial-wise evaluation
        ttf_trial = []   # for trial-wise evaluation

        for sub in subject:
            data, label = self.load_per_subject(sub)
            va = Averager()
            va_val = Averager()
            preds, acts = [], []
            preds_trial, acts_trial = [], []
            kf = KFold(n_splits=fold, shuffle=shuffle)
            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label)

                data_train, label_train, data_val, label_val = self.split_balance_class(
                    data=data_train, label=label_train, train_rate=self.args.training_rate, random=True
                )

                if reproduce:
                    # to reproduce the reported ACC
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=reproduce,
                                               subject=sub, fold=idx_fold)
                    acc_val = 0
                else:
                    # to train new models
                    # Check the dimension of the training, validation and test set
                    print('Training:', data_train.size(), label_train.size())
                    print('Validation:', data_val.size(), label_val.size())
                    print('Test:', data_test.size(), label_test.size())

                    acc_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=sub,
                                    fold=idx_fold)
                    # test the model on testing data
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=reproduce,
                                               subject=sub, fold=idx_fold)

                # get trial-wise prediction
                act_trial, pred_trial = self.trial_wise_voting(
                    act=act, pred=pred,
                    num_segment_per_trial=int(self.args.trial_duration/self.args.segment),
                    trial_in_fold=len(idx_test)
                )
                va_val.add(acc_val)
                va.add(acc_test)
                preds.extend(pred)
                acts.extend(act)
                preds_trial.extend(pred_trial)
                acts_trial.extend(act_trial)

            tva.append(va_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            acc_trial, f1_trial, _ = get_metrics(y_pred=preds_trial, y_true=acts_trial)

            tta.append(acc)
            ttf.append(f1)
            tta_trial.append(acc_trial)
            ttf_trial.append(f1_trial)
            result = '{},{}'.format(tta[-1], ttf[-1])
            self.log2txt(result)

        # prepare final report
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)

        mACC_trial = np.mean(tta_trial)
        mF1_trial = np.mean(ttf_trial)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        results = 'test mAcc={} mF1={} val mAcc={}'.format(mACC, mF1, mACC_val)
        self.log2txt(results)
        results = 'test mAcc={} mF1={} (trial-wise)'.format(mACC_trial, mF1_trial)
        self.log2txt(results)

    def log2txt(self, content):
        """
        This function log the content to results.txt
        param content: string, the content to log.
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()

    def trial_wise_voting(self, act, pred, num_segment_per_trial, trial_in_fold):
        """
        this function does voting within each tiral to get the label of entire trial
        param act: [num_sample] list
        param pred: [num_sample] list
        param num_segment_per_trial: how many samples per trial
        param trial_in_fold: how many trials in this fold
        return: trial-wise actual label and predicted label.
        """
        num_trial = int(len(act)/num_segment_per_trial)
        assert num_trial == trial_in_fold
        act_trial = np.reshape(act, (num_trial, num_segment_per_trial))
        pred_trial = np.reshape(pred, (num_trial, num_segment_per_trial))

        act_trial = np.mean(act_trial, axis=-1).tolist()
        pred_vote = []
        for trial in pred_trial:
            index_0 = np.where(trial == 0)[0]
            index_1 = np.where(trial == 1)[0]
            if len(index_1) >= len(index_0):
                label = 1
            else:
                label = 0
            pred_vote.append(label)
        return act_trial, pred_vote
