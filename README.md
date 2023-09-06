

# QLoRA for Masked Language Modeling

This repo updates [QLoRA](https://github.com/artidoro/qlora) for use with the Masked Language Modeling objective, which is commonly used in encoder-decoder architectures like BERT and its derivatives. With this repo, large MLM models can now be finetuned comfortably on consumer hardware.

## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library. After installing PyTorch (follow instructions [here](https://pytorch.org/get-started/locally/)), you can achieve the above with the following command:
```bash
pip install -U -r requirements.txt
```

## Getting Started
The `qlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on a local dataset:
```bash
python qlora.py --model_name_or_path <path_or_name> --dataset <relative_path_to_dataset>
```

For models larger than 13B, we recommend adjusting the learning rate:
```bash
python qlora.py –learning_rate 0.0001 --model_name_or_path <path_or_name>
```

Examples are found under the `examples/` folder.

### Quantization
Quantization parameters are controlled from the `BitsandbytesConfig` ([see HF documenation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)) as follows:
- Loading in 4 bits is activated through `load_in_4bit`
- The datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- Nested quantization is activated through `bnb_4bit_use_double_quant`
- The datatype used for qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using `nf4`.

```python
    model = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
```

### Paged Optimizer
You can access the paged optimizer with the argument `--optim paged_adamw_32bit`

### Using Local Datasets

You can specify the path to your dataset using the `--dataset` argument.

### Multi GPU
Multi GPU training and inference work out-of-the-box with Hugging Face's Accelerate. Note that the `per_device_train_batch_size` and `per_device_eval_batch_size` arguments are  global batch sizes unlike what their name suggest.

When loading a model for training or inference on multiple GPUs you should pass something like the following to `AutoModelForCausalLM.from_pretrained()`:
```python
device_map = "auto"
max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
```

## Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```