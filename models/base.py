from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):

    @abstractmethod
    def build(self, input_classes, output_classes):
        pass

    @abstractmethod
    def call(self, inputs, training=False):
        pass
