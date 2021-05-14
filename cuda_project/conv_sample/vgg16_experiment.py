from experiment import Experiment


def vgg16_conv1_batch_size_experiment(name="vgg16_conv1_batch", grad='', filterFormat=0):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256]
    exp.params['c'] = [64]
    exp.params['h'] = [224]
    exp.params['w'] = [224]
    exp.params['k'] = [64]
    exp.params['r'] = [3]
    exp.params['s'] = [3]
    exp.params['pad_w'] = [1]
    exp.params['pad_h'] = [1]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    # exp.params['dataType'] = [0, 1, 2, 3]
    exp.params['filterFormat'] = filterFormat
    exp.grad = grad
    exp.run()
    exp.export_to_csv()


def vgg16_conv2_batch_size_experiment(name="vgg16_conv2_batch", grad='', filterFormat=0):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256, 512]
    exp.params['c'] = [128]
    exp.params['h'] = [112]
    exp.params['w'] = [112]
    exp.params['k'] = [128]
    exp.params['r'] = [3]
    exp.params['s'] = [3]
    exp.params['pad_w'] = [1]
    exp.params['pad_h'] = [1]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.params['filterFormat'] = filterFormat
    exp.grad = grad
    exp.run()
    exp.export_to_csv()


def vgg16_conv3_batch_size_experiment(name="vgg16_conv3_batch", grad='', filterFormat=0):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256, 512, 1024]
    exp.params['c'] = [256]
    exp.params['h'] = [56]
    exp.params['w'] = [56]
    exp.params['k'] = [256]
    exp.params['r'] = [3]
    exp.params['s'] = [3]
    exp.params['pad_w'] = [1]
    exp.params['pad_h'] = [1]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.params['filterFormat'] = filterFormat
    exp.grad = grad
    exp.run()
    exp.export_to_csv()


def vgg16_conv4_batch_size_experiment(name="vgg16_conv4_batch", grad='', filterFormat=0):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    exp.params['c'] = [512]
    exp.params['h'] = [28]
    exp.params['w'] = [28]
    exp.params['k'] = [512]
    exp.params['r'] = [3]
    exp.params['s'] = [3]
    exp.params['pad_w'] = [1]
    exp.params['pad_h'] = [1]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.params['filterFormat'] = filterFormat
    exp.grad = grad
    exp.run()
    exp.export_to_csv()


def vgg16_conv5_batch_size_experiment(name="vgg16_conv5_batch", grad='', filterFormat=0):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    exp.params['c'] = [512]
    exp.params['h'] = [14]
    exp.params['w'] = [14]
    exp.params['k'] = [512]
    exp.params['r'] = [3]
    exp.params['s'] = [3]
    exp.params['pad_w'] = [1]
    exp.params['pad_h'] = [1]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.params['filterFormat'] = filterFormat
    exp.grad = grad
    exp.run()
    exp.export_to_csv()


def vgg16_fc_batch_size_experiment(name="vgg16_fc_batch", grad='', filterFormat=0):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    exp.params['c'] = [4096]
    exp.params['h'] = [1]
    exp.params['w'] = [1]
    exp.params['k'] = [4096]
    exp.params['r'] = [1]
    exp.params['s'] = [1]
    exp.params['pad_w'] = [0]
    exp.params['pad_h'] = [0]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.params['filterFormat'] = filterFormat
    exp.grad = grad
    exp.run()
    exp.export_to_csv()


def main():

    # vgg16_conv1_batch_size_experiment()
    # vgg16_conv2_batch_size_experiment()
    # vgg16_conv3_batch_size_experiment()
    # vgg16_conv4_batch_size_experiment()
    # vgg16_conv5_batch_size_experiment()
    # vgg16_fc_batch_size_experiment()

    vgg16_conv1_batch_size_experiment(name='vgg16_conv1_dgrad', grad='-dgrad', filterFormat=[0])
    vgg16_conv2_batch_size_experiment(name='vgg16_conv2_dgrad', grad='-dgrad', filterFormat=[0])
    vgg16_conv3_batch_size_experiment(name='vgg16_conv3_dgrad', grad='-dgrad', filterFormat=[0])
    vgg16_conv4_batch_size_experiment(name='vgg16_conv4_dgrad', grad='-dgrad', filterFormat=[0])
    vgg16_conv5_batch_size_experiment(name='vgg16_conv5_dgrad', grad='-dgrad', filterFormat=[0])
    vgg16_fc_batch_size_experiment(name='vgg16_fc_dgrad', grad='-dgrad', filterFormat=[0])


if __name__ == "__main__":
    main()
