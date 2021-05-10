import subprocess
import itertools
import csv


class Experiment:

    def __init__(self, name, output_dir="./results", mathType=0, dataType=0,
                 n=1, c=512, h=64, w=64, k=64, r=8, s=8, pad_h=0, pad_w=0, u=1, v=1):

        # Name this set of experiments, used for file export and readability
        self.name = name
        self.output_dir = output_dir

        # Store execution of an experiment
        self.runs = []

        # Set up parameter store
        self.params = {}
        self.params['mathType'] = [mathType]
        self.params['dataType'] = [dataType]
        self.params['n'] = [n]
        self.params['c'] = [c]
        self.params['h'] = [h]
        self.params['w'] = [w]
        self.params['k'] = [k]
        self.params['r'] = [r]
        self.params['s'] = [s]
        self.params['pad_h'] = [pad_h]
        self.params['pad_w'] = [pad_w]
        self.params['u'] = [u]
        self.params['v'] = [v]

        # Assume these values when doing mass experiments for data export
        self.verbose = False
        self.benchmark = True
        
    def assemble_command(self, experiment):    
        command = './conv_sample'
        try:
            if experiment['dataType'] > 1:
                experiment['filterFormat'] = 2
                assert experiment['mathType'] == 1, "INT8 requires filter format 2"
            else:
                experiment['filterFormat'] = 0
            assert (experiment['mathType'] == 1 and experiment['dataType'] == 0) != True, "Cannot use TC for FP33"
            for k, v in experiment.items():
                command += ' -{}{}'.format(k, v)
            if not self.verbose:
                command += ' -e'
            if self.benchmark:
                command += ' -b'
            return command
        except:
            return None

    def run(self):
        print(f"Running Experiment <{self.name}>...")
        keys, vals = zip(*self.params.items())
        for val in itertools.product(*vals):
            experiment = dict(zip(keys, val))
            command = self.assemble_command(experiment)
            if command is None:
                continue
            else:
                print(command)
                output = subprocess.check_output(command, shell=True, encoding='UTF-8').rstrip()
                experiment['result'] = float(output)
                self.runs.append(experiment)

    def export_to_csv(self):
        filename = "{}/{}_results.csv".format(self.output_dir, self.name)
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ' ', quotechar = '|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(self.runs[0].keys())
            for run in self.runs:
                csvwriter.writerow(run.values())
        
def main():
    base = Experiment("base", n=32)
    base.params['mathType'] = [0, 1]
    base.params['dataType'] = [0, 1, 2, 3]
    base.run()
    base.export_to_csv()

if __name__ == "__main__":
    main()
