import dsdtools
from consts import CONSTS
import processing
import model_constructor
import gc
import keras


class Pipeline(object):

    def __init__(self, root_dir=CONSTS.DB_PATH, targets=CONSTS.TARGETS):
        self.DB = dsdtools.DB(root_dir=root_dir )
        self.dev_set = self.DB.load_dsd_tracks(subsets='Dev')
        self.test_set = self.DB.load_dsd_tracks(subsets='Test')
        self.targets = targets
        self.models = [model_constructor.make_compile_model(target) for target in self.targets]
        self.models_input = processing.make_input(self.dev_set)

    def train_model(self, target, epochs=10, batch_size=16):
        if target not in self.targets:
            raise ValueError('Unexpected target')
        model = self.models[self.targets.index(target)]
        model_target = processing.make_target_data(self.dev_set, target)
        checkpointer = keras.callbacks.ModelCheckpoint(model.name + '_DNN_model', monitor='val_loss', verbose=0,
                                                       save_best_only=False, save_weights_only=False, mode='auto', period=1)
        model.fit(self.models_input, model_target,  epochs=epochs, batch_size=batch_size, callbacks = [checkpointer])
        del model_target; gc.collect()

    def train_models(self, epochs=10, batch_size=16):
        for target in self.targets:
            self.train_model(target, epochs, batch_size)

    def single_track_estimations(self, track):
        estimations = {}
        for counter, target in self.targets:
            estimations[target] = processing.make_predictions(self.models[counter], track, target)
        processing.save_estimates(estimations, track, estimates_dir=CONSTS.ESTIMATES_PATH)

    def run_estimations(self):
        for track in self.test_set:
            self.single_track_estimations(track)


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.train_models(epochs=100, batch_size=16)
    pipeline.run_estimations()
