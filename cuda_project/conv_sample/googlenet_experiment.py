from experiment import Experiment


def inception_module_test(name="inception_3A", n=[1], c_in=192, h_in=28,
        w_in=28, n_1x1=64, n_3x3R=96, n_3x3=128, n_5x5R=16, n_5x5=32, pool_proj=32,
        configs = [(0,0)], grad=''):
   
    exp = Experiment(name)
    exp.grad = grad

    for mathType, dataType in configs:
        for n_in in n:

            runtime = 0

            # Unchanging params for each conv
            exp.params['n'] = [n_in]
            exp.params['h'] = [h_in]
            exp.params['w'] = [w_in]
            exp.params['pad_w'] = [1]
            exp.params['pad_h'] = [1]
            exp.params['u'] = [1]
            exp.params['v'] = [1]
            exp.params['mathType'] = [mathType]
            exp.params['dataType'] = [dataType]

            # 1x1 Conv
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_1x1_conv"
            exp.params['c'] = [c_in]
            exp.params['r'] = [1]
            exp.params['s'] = [1]
            exp.params['k'] = [n_1x1]
            runtime += exp.run_single()

            # 3x3 Reduction
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_3x3_conv_reduction"
            exp.params['c'] = [c_in]
            exp.params['r'] = [1]
            exp.params['s'] = [1]
            exp.params['k'] = [n_3x3R]
            runtime += exp.run_single()

            # 3x3 Conv
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_3x3_conv"
            exp.params['c'] = [n_3x3R]
            exp.params['r'] = [3]
            exp.params['s'] = [3]
            exp.params['k'] = [n_3x3]
            runtime += exp.run_single()

            # 5x5 Reduction
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_5x5_conv_reduction"
            exp.params['c'] = [c_in]
            exp.params['r'] = [1]
            exp.params['s'] = [1]
            exp.params['k'] = [n_5x5R]
            runtime += exp.run_single()

            # 5x5 Conv
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_5x5_conv"
            exp.params['c'] = [n_5x5R]
            exp.params['r'] = [5]
            exp.params['s'] = [5]
            exp.params['k'] = [n_5x5]
            runtime += exp.run_single()

            # 1x1 Pool Conv
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_1x1_pool_conv"
            exp.params['c'] = [c_in]
            exp.params['r'] = [1]
            exp.params['s'] = [1]
            exp.params['k'] = [pool_proj]
            runtime += exp.run_single()

            current = {}
            current["name"] = name
            current["mathType"] = mathType
            current["dataType"] = dataType
            current["N"] = n_in
            current["runtime"] = runtime
            exp.runs.append(current) 

    exp.name = name
    exp.export_to_csv()


def run_inception_tests(grad=''):
    
    n = [1, 16, 32, 64, 128, 256]
    configs = [(0,0),(0,1),(1,1)]

    inception_module_test(name="inception_3A", n=n, c_in=192, h_in=28,
        w_in=28, n_1x1=64, n_3x3R=96, n_3x3=128, n_5x5R=16, n_5x5=32, pool_proj=32,
        configs=configs, grad=grad)

    inception_module_test(name="inception_3B", n=n, c_in=256, h_in=28,
        w_in=28, n_1x1=128, n_3x3R=128, n_3x3=192, n_5x5R=32, n_5x5=96, pool_proj=64,
        configs=configs, grad=grad)

    inception_module_test(name="inception_4A", n=n, c_in=480, h_in=14,
        w_in=14, n_1x1=192, n_3x3R=96, n_3x3=208, n_5x5R=16, n_5x5=48, pool_proj=64,
        configs=configs, grad=grad)

    inception_module_test(name="inception_4B", n=n, c_in=512, h_in=14,
        w_in=14, n_1x1=160, n_3x3R=112, n_3x3=224, n_5x5R=24, n_5x5=64, pool_proj=64,
        configs=configs, grad=grad)

    inception_module_test(name="inception_4C", n=n, c_in=512, h_in=14,
        w_in=14, n_1x1=128, n_3x3R=128, n_3x3=256, n_5x5R=24, n_5x5=64, pool_proj=64,
        configs=configs, grad=grad)

    inception_module_test(name="inception_4D", n=n, c_in=512, h_in=14,
        w_in=14, n_1x1=112, n_3x3R=144, n_3x3=288, n_5x5R=32, n_5x5=64, pool_proj=64,
        configs=configs, grad=grad)

    inception_module_test(name="inception_4E", n=n, c_in=528, h_in=14,
        w_in=14, n_1x1=256, n_3x3R=160, n_3x3=320, n_5x5R=32, n_5x5=128, pool_proj=128,
        configs=configs, grad=grad)
 
    inception_module_test(name="inception_5A", n=n, c_in=832, h_in=7,
        w_in=7, n_1x1=256, n_3x3R=160, n_3x3=320, n_5x5R=32, n_5x5=128, pool_proj=128,
        configs=configs, grad=grad)

    inception_module_test(name="inception_5B", n=n, c_in=832, h_in=7,
        w_in=7, n_1x1=384, n_3x3R=192, n_3x3=384, n_5x5R=48, n_5x5=128, pool_proj=128,
        configs=configs, grad=grad)

def googlenet_conv1(name='googlenet_conv1', grad=''):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256]
    exp.params['c'] = [3]
    exp.params['h'] = [224]
    exp.params['w'] = [224]
    exp.params['k'] = [64]
    exp.params['r'] = [7]
    exp.params['s'] = [7]
    exp.params['u'] = [2]
    exp.params['v'] = [2]
    exp.params['pad_w'] = [2]
    exp.params['pad_h'] = [2]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.grad = grad
    exp.run()
    exp.export_to_csv()

def googlenet_conv2(name='googlenet_conv2', grad=''):

    n = [1, 16, 32, 64, 128, 256]
    configs = [(0,0),(0,1),(1,1)]
    exp = Experiment(name)
    exp.grad = grad

    for mathType, dataType in configs:
        for n_in in n:

            runtime = 0

            # Unchanging params for each conv
            exp.params['n'] = [n_in]
            exp.params['h'] = [56]
            exp.params['w'] = [56]
            exp.params['pad_w'] = [1]
            exp.params['pad_h'] = [1]
            exp.params['u'] = [1]
            exp.params['v'] = [1]
            exp.params['mathType'] = [mathType]
            exp.params['dataType'] = [dataType]

            # 3x3 Reduction
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_3x3_conv_reduction"
            exp.params['c'] = [64]
            exp.params['r'] = [1]
            exp.params['s'] = [1]
            exp.params['k'] = [64]
            runtime += exp.run_single()

            # 3x3 Conv
            exp.name = name + "_m" + str(mathType) + "_d" + str(dataType) + "_3x3_conv"
            exp.params['c'] = [64]
            exp.params['r'] = [3]
            exp.params['s'] = [3]
            exp.params['k'] = [192]
            runtime += exp.run_single()

            current = {}
            current["name"] = name
            current["mathType"] = mathType
            current["dataType"] = dataType
            current["N"] = n_in
            current["runtime"] = runtime
            exp.runs.append(current) 

    exp.name = name
    exp.export_to_csv()

def run_convolution_tests(grad=''):
    googlenet_conv1(name="googlenet_conv1_" + grad, grad=grad)
    googlenet_conv2(name="googlenet_conv2_" + grad, grad=grad)

def run_googlenet_fc(name="googlenet_fc", grad=''):
    
    exp = Experiment(name)
    exp.params['n'] = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    exp.params['c'] = [1024]
    exp.params['h'] = [1]
    exp.params['w'] = [1]
    exp.params['k'] = [1000]
    exp.params['r'] = [1]
    exp.params['s'] = [1]
    exp.params['pad_w'] = [0]
    exp.params['pad_h'] = [0]
    exp.params['mathType'] = [0, 1]
    exp.params['dataType'] = [0, 1]
    exp.run()
    exp.export_to_csv()


def main():
    # run_inception_tests()
    # run_convolution_tests()
    # run_googlenet_fc()
    run_inception_tests(grad='-dgrad')
    run_convolution_tests(grad='-dgrad')
    run_googlenet_fc(grad='-dgrad')

if __name__ == "__main__":
    main()
