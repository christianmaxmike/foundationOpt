import argparse
from generation import SimpleTrig_1D, SineCosGenerator

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="A script to configure and run a model with various hyperparameters."
    )

    parser.add_argument(
        "--id",
        default=0,
        type=int,
        help="An integer identifier for the current run or experiment. Default is 0."
    )

    gen_models = ["SimpleTrig", "SineCos", "MLP"]
    parser.add_argument(
        "--type",
        default="SineCos",
        choices=gen_models,
        help=f"Type of generator {gen_models} to be used to synthesize function. Default is 'SineCos'."
    )

    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "multi"],
        help="Mode being use for generation. single:1D vs. multi:[minDim; maxDim]."
    )

    parser.add_argument(
        "--batchSize",
        default=4096,
        type=int,
        help="The size of each batch used during training or processing. Default is 8192."
    )

    parser.add_argument(
        "--sequenceLength",
        default=100,
        type=int,
        help="The length of the input sequences to be processed. Default is 100."
    )

    parser.add_argument(
        "--minComponents",
        default=2,
        type=int,
        help="The minimum number of components to use in the model. Default is 2."
    )

    parser.add_argument(
        "--maxComponents",
        default=5,
        type=int,
        help="The maximum number of components to use in the model. Default is 8."
    )

    parser.add_argument(
        "--minX",
        default=0,
        type=int,
        help="The minimum value for the range of input data (e.g., x-axis). Default is 0."
    )

    parser.add_argument(
        "--maxX",
        default=1,
        type=int,
        help="The maximum value for the range of input data (e.g., x-axis). Default is 1."
    )

    parser.add_argument(
        "--numHeboTrials",
        default=10,
        type=int,
        help="The number of trials to run during HEBO (Heteroscedastic Evolutionary Bayesian Optimization). Default is 10."
    )

    parser.add_argument(
        "--numHeboSteps",
        default=50,
        type=int,
        help="The number of optimization steps to perform during HEBO. Default is 100."
    )

    parser.add_argument(
        "--minDim",
        default=1,
        type=int,
        help="The minimum dimension of generated functions. Default is 1."
    )

    parser.add_argument(
        "--maxDim",
        default=10,
        type=int,
        help="The maximum dimension of generated functions. Default is 10."
    )

    # Parse arguments
    args = parser.parse_args()

    model  = None
    if args.type == "SimpleTrig":
        model = SimpleTrig_1D
    elif args.type == "SineCos":
        model = SineCosGenerator
    else:
        raise Exception("Other models currently not tested in new framework. to be integrated asap.")
    
    print("Generating functions...")
    model.generate(args)
