from mnist import MNIST


class MNISTData(MNIST):
    def __init__(self, path: str):
        super().__init__(path)
        self.gz = True

    def get_test_data(self):
        test_images, test_labels = self.load_testing()
        return test_images, test_labels

    def get_train_data(self):
        train_images, train_labels = self.load_training()
        return  train_images, train_labels

