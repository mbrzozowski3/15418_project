from experiment import Experiment


def base_experiment():

    exp = Experiment("base")
    exp.run()
    exp.export_to_csv()


def main():

    base_experiment()


if __name__ == "__main__":
    main()
