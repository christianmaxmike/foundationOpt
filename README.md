# Foundation Optimization

Foundation Optimization is a Python-based framework designed to synthesize mathematical functions and train a transformer module to learn optimization steps. This project is particularly useful for exploring function generation and optimization techniques using machine learning models.

## Features
- **Function Generation**: Generate functions using predefined models such as `SimpleTrig` and `SineCos`.
- **Transformer Training**: Train a transformer module to learn optimization strategies.
- **Customizable Parameters**: Configure dimensions, batch size, sequence length, and more through command-line arguments.
- **HEBO Integration**: Perform Heteroscedastic Evolutionary Bayesian Optimization with configurable trials and steps.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/foundation-optimization.git
cd foundation-optimization
```


2. Install dependencies:

```bash
pip install -r requirements.txt
```



---

## Usage

The entry point for the program is the script `main.py`. The script allows you to configure various hyperparameters for function generation and optimization.

### Command-Line Arguments

| Argument            | Default Value | Description                                                                 |
|---------------------|---------------|-----------------------------------------------------------------------------|
| `--id`             | `0`           | Integer identifier for the current run or experiment.                       |
| `--type`           | `"SimpleTrig"`| Type of generator model (`SimpleTrig`, `SineCos`, `MLP`).                   |
| `--mode`           | `"single"`    | Mode for generation (`single`: 1D, `multi`: [minDim; maxDim]).              |
| `--batchSize`      | `8192`        | Batch size used during training or processing.                              |
| `--sequenceLength` | `100`         | Length of input sequences to process.                                       |
| `--minComponents`  | `2`           | Minimum number of components in the model.                                  |
| `--maxComponents`  | `8`           | Maximum number of components in the model.                                  |
| `--minX`           | `0`           | Minimum value for the range of input data (e.g., x-axis).                   |
| `--maxX`           | `1`           | Maximum value for the range of input data (e.g., x-axis).                   |
| `--numHeboTrials`  | `10`          | Number of trials during HEBO optimization.                                  |
| `--numHeboSteps`   | `100`         | Number of optimization steps during HEBO.                                   |
| `--minDim`         | `1`           | Minimum dimension of generated functions.                                   |
| `--maxDim`         | `10`          | Maximum dimension of generated functions.                                   |

---

### Example: Generating Functions

To generate functions using the default configuration, run:

```bash
python main.py
```


To specify a custom generator type (e.g., SineCosGenerator) and adjust parameters:

```bash
python main.py --type SineCos --batchSize 4096 --sequenceLength 50 --minX -1 --maxX 1
```


### Output

The script will output files including the hebo steps in the folder: `datasets/[single|multi]/[1D|min_d-max_d]/<files>`


---

## Sample Code

Below is an example of how to use the function generators directly in Python:


```python
from generation import SimpleTrig_1D, SineCosGenerator

# Configure parameters
args = {
    "batchSize": 4096,
    "sequenceLength": 50,
    "minX": -1,
    "maxX": 1,
    "minComponents": 3,
    "maxComponents": 5,
}

# Use SimpleTrig generator
generator = SimpleTrig_1D()
generator.generate(args)

# Use SineCos generator
generator = SineCosGenerator()
generator.generate(args)
```



---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy optimizing! 🚀
