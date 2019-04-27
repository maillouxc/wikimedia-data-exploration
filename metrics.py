import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        _val_acc = accuracy_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)

        self.val_accs.append(_val_acc)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print(f'- accuracy: {_val_acc} — precision: {_val_precision} — recall {_val_recall} — f1: {_val_f1}')
        return

    @staticmethod
    def get_csv_header(run_title, f1_type):
        return str.join(', ', [
            f'{run_title}',
            'Accuracy',
            'Precision',
            'Recall',
            f'{f1_type}'
        ])
