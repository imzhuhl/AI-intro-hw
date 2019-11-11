

class Config:
    load_weight = False
    use_cuda = True
    learning_rate = 0.001
    num_epoch = 200
    num_classes = 6

    data_path = "./dataset-resized/"
    weight_path = "./weights/"
    classes = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
