import abc

from singleton import Singleton


class ContainerRegisterMetaClass(abc.ABCMeta):
    def __init__(cls, name, bases, attr):
        Container().register(name, cls)


class Container(object):
    """
    Dependency injection class for distributing implementations across the application, use register to add a new type
    to the container and create to create an instance of that type, each type is saved with a unique string key
    """
    __metaclass__ = Singleton

    def __init__(self):
        self.items = {}

    def register(self, name, type):
        self.items[name] = type

    def create(self, name, *args):
        item_type = self.items[name]
        return item_type(*args)
