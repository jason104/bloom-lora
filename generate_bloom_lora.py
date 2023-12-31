import torch
from peft import PeftModel
import transformers
#import gradio as gr

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, GenerationConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer

import json
import glob


input_dir = './yc_eval/**/*.jsonl'
input_files = glob.glob(input_dir, recursive=True)
input_all = {}

for filename in input_files:
    with open(filename, 'r') as json_file:
        json_list = list(json_file)
    input_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        input_list.append(result['prompt'])
    input_all[filename] = input_list
    #print(result['prompt'])
    #print(f"result: {result}")
    #print(isinstance(result, dict))

tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom')

#BASE_MODEL = "bigscience/bloom-3b"
BASE_MODEL = "bigscience/bloom-7b1"

#LORA_WEIGHTS = "LinhDuong/bloom-7b1-alpaca"
LORA_WEIGHTS = "bloom-7b1-alpaca-lora"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#try:
#    if torch.backends.mps.is_available():
#        device = "mps"
#except:
#    pass

if device == "cuda":
    model = BloomForCausalLM.from_pretrained(
        BASE_MODEL,
        #load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, dtype=torch.float16)
    #model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
elif device == "mps":
    model = BloomForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = BloomForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        max_new_tokens=512,
        repetition_penalty=3.5,
        temperature=0.5,
        top_p=1,
        top_k=50,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


#gr.Interface(
#    fn=evaluate,
#    inputs=[
#        gr.components.Textbox(
#            lines=2, label="Instruction", placeholder="Tell me about alpacas."
#        ),
#        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
#        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
#        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
#        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
#        gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
#        gr.components.Slider(
#            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
#        ),
#    ],
#    outputs=[
#        gr.inputs.Textbox(
#            lines=5,
#            label="Output",
#        )
#    ],
#    title="🌲 🌲 🌲 BLOOM-LoRA",
#    description="BLOOM-LoRA is a 560M-parameter BLOOM model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",
#).launch()

# Old testing code follows.


if __name__ == "__main__":


    # testing code for readme
    #for instruction in [
    #    "Tell me about alpacas.",
    #    "Tell me about the president of Mexico in 2019.",
    #    "Tell me about the king of France in 2019.",
    #    "List all Canadian provinces in alphabetical order.",
    #    "Write a Python program that prints the first 10 Fibonacci numbers.",
    #    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    #    "Tell me five words that rhyme with 'shock'.",
    #    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #    "Count up from 1 to 500.",
    #]:
        #print("Instruction:", instruction)
        #print("Response:", evaluate(instruction))
        #print()

    for filename, input_list in input_all.items():
        output_list = []
        for instruction in input_list:
            output = evaluate(instruction)
            output = {'prompt': instruction, 'ycb7bca': output}
            output_list.append(output)
        #print("Response:", output)
        #print()
    #print(output_list)

        with open(filename[:-6] + '_ycb7bca.jsonl', 'w', encoding='utf8') as outfile:
            for entry in output_list:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')

