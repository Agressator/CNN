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
        adapt_train_labels = list()
        for label in train_labels:
            adapt_label = [0 for i in range(10)]
            adapt_label[label] = 1
            adapt_train_labels.append(adapt_label)

        return train_images, adapt_train_labels

