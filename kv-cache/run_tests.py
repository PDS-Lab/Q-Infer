import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0

run_baseline = True
run_InfiniGen = False
run_H2O = True
run_Keyformer = False
run_Qinfer = True
run_Quant = False


# opt_sizes = ["6.7b", "13b", "30b"]
opt_sizes = ["6.7b"]
# run_tasks = ["copa", "openbookqa"]
run_tasks = ["winogrande"]
#run_tasks = ["piqa", "hellaswag"]
run_llama_2 = False

# h2o_ratios = [0.25, 0.125, 0.0625, 0.03125]
# h2o_ratios = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
h2o_ratios = [0.05]

# Prepare dataset
# piqa 似乎有问题 utf-8 error
#for task in ["piqa", "openbookqa", "winogrande", "copa", "rte"]:
for task in run_tasks:
    cmd = []
    cmd.append("python -u generate_task_data.py")
    cmd.append(f"--output-file results/{task}-{shots}.jsonl")
    cmd.append(f"--task-name {task}")
    cmd.append(f"--num-fewshot {shots}")
    cmd = ' '.join(cmd)
    os.system(cmd)

## Baseline
if run_baseline:
    print("run_baseline")
    print("="*10+" Full cache " + "="*10)
    # OPT
    for size in opt_sizes:
        if size == "6.7b":
            # tasks = ["piqa", "openbookqa"]
            tasks = run_tasks
        elif size == "13b":
            tasks = ["winogrande", "openbookqa"]
        elif size == "30b":
            tasks = ["copa", "openbookqa"]
        for task in tasks:
            cmd = []
            cmd.append("bash full_cache.sh")
            cmd.append(task)
            cmd.append(f"facebook/opt-{size}")
            cmd.append("opt")
            cmd.append(str(shots))
            cmd = ' '.join(cmd)
            print(cmd)
            os.system(cmd)
            print("-------------------------------------------")

    # Llama-2
    if run_llama_2:
        llama_2_dir = os.environ["LLAMA_PATH"]
        for size in ["7b", "13b"]:
            if size == "7b":
                tasks = ["rte", "piqa"]
            elif size == "13b":
                tasks = ["copa", "winogrande"]
            for task in tasks:
                cmd = []
                cmd.append("bash full_cache.sh")
                cmd.append(task)
                cmd.append(f"{llama_2_dir}/llama-2-{size}")
                cmd.append("llama")
                cmd.append(str(shots))
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")


## InfiniGen
if run_InfiniGen:
    print("="*10+" InfiniGen " + "="*10)
    # OPT
    for size in opt_sizes:
        if size == "6.7b":
            # tasks = ["piqa", "openbookqa"]
            tasks = run_tasks
        elif size == "13b":
            # tasks = ["winogrande", "openbookqa"]
            tasks = run_tasks
        elif size == "30b":
            tasks = ["copa", "openbookqa"]
        for task in tasks:
            for retain_ratio in range(4):
                alpha, budget = config[f"opt-{size}"][task][retain_ratio]
                cmd = []
                cmd.append("bash ours.sh")
                cmd.append(task)
                cmd.append(f"../setup/opt-model/opt-{size}")
                cmd.append(f"facebook/opt-{size}")
                cmd.append("opt")
                cmd.append(str(shots))
                cmd.append(str(partial))
                cmd.append(str(alpha))
                cmd.append(str(capacity))
                cmd.append(str(budget))
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

    # Llama-2
    if run_llama_2:
        llama_2_dir = os.environ["LLAMA_PATH"]
        for size in ["7b", "13b"]:
            if size == "7b":
                tasks = ["rte", "piqa"]
            elif size == "13b":
                tasks = ["copa", "winogrande"]
            for task in tasks:
                for retain_ratio in range(4):
                    alpha, budget = config[f"llama-2-{size}"][task][retain_ratio]
                    cmd = []
                    cmd.append("bash ours.sh")
                    cmd.append(task)
                    cmd.append(f"{llama_2_dir}/llama-2-{size}")
                    cmd.append(f"{llama_2_dir}/llama-2-{size}")
                    cmd.append("llama")
                    cmd.append(str(shots))
                    cmd.append(str(partial))
                    cmd.append(str(alpha))
                    cmd.append(str(capacity))
                    cmd.append(str(budget))
                    cmd = ' '.join(cmd)
                    print(cmd)
                    os.system(cmd)
                    print("-------------------------------------------")


## H2O
if run_H2O:
    print("="*10+" H2O " + "="*10)
    # OPT
    for size in opt_sizes:
        if size == "6.7b":
            # tasks = ["piqa", "openbookqa"]
            tasks = run_tasks
        elif size == "13b":
            tasks = ["winogrande", "openbookqa"]
        elif size == "30b":
            tasks = ["copa", "openbookqa"]
        for task in tasks:
            for ratio in h2o_ratios:
                cmd = []
                cmd.append("bash h2o.sh")
                cmd.append(task)
                cmd.append(f"facebook/opt-{size}")
                cmd.append("opt")
                cmd.append(str(shots))
                cmd.append(str(ratio)) # heavy_ratio
                cmd.append(str(ratio)) # recent_ratio
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

    # Llama-2
    if run_llama_2:
        llama_2_dir = os.environ["LLAMA_PATH"]
        for size in ["7b", "13b"]:
            if size == "7b":
                tasks = ["rte", "piqa"]
            elif size == "13b":
                tasks = ["copa", "winogrande"]
            for task in tasks:
                for ratio in [0.25, 0.125, 0.0625, 0.03125]:
                    cmd = []
                    cmd.append("bash h2o.sh")
                    cmd.append(task)
                    cmd.append(f"{llama_2_dir}/llama-2-{size}")
                    cmd.append("llama")
                    cmd.append(str(shots))
                    cmd.append(str(ratio)) # heavy_ratio
                    cmd.append(str(ratio)) # recent_ratio
                    cmd = ' '.join(cmd)
                    print(cmd)
                    os.system(cmd)
                    print("-------------------------------------------")

if run_Keyformer:
    print("="*10+" KeyFormer " + "="*10)
    # OPT
    for size in opt_sizes:
        if size == "6.7b":
            tasks = run_tasks
        for task in tasks:
            for ratio in h2o_ratios:
                ratio = ratio * 2
                recent_ratio = ratio * 0.3
                heavy_ratio = ratio - recent_ratio

                cmd = []
                cmd.append("bash keyformer.sh")
                cmd.append(task)
                cmd.append(f"facebook/opt-{size}")
                cmd.append("opt")
                cmd.append(str(shots))
                cmd.append(str(heavy_ratio)) # heavy_ratio
                cmd.append(str(recent_ratio)) # recent_ratio
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

    # Llama-2

if run_Qinfer:
    print("="*10+" Qinfer " + "="*10)
    # OPT
    for size in opt_sizes:
        if size == "6.7b":
            tasks = run_tasks
        for task in tasks:
            for ratio in h2o_ratios:
                ratio = ratio * 2
                recent_ratio = ratio * 0.5
                heavy_ratio = ratio - recent_ratio

                cmd = []
                cmd.append("bash qinfer.sh")
                cmd.append(task)
                cmd.append(f"facebook/opt-{size}")
                cmd.append("opt")
                cmd.append(str(shots))
                cmd.append(str(heavy_ratio)) # heavy_ratio
                cmd.append(str(recent_ratio)) # recent_ratio
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

    # Llama-2

## Quant.
if run_Quant:
    print("="*10+" Quantization " + "="*10)
    # OPT
    for size in opt_sizes:
        if size == "6.7b":
            # tasks = ["piqa", "openbookqa"]
            tasks = run_tasks
        elif size == "13b":
            tasks = ["winogrande", "openbookqa"]
        elif size == "30b":
            tasks = ["copa", "openbookqa"]
        for task in tasks:
            for qbits in [8, 4, 2, 1]:
                cmd = []
                cmd.append("bash quant.sh")
                cmd.append(task)
                cmd.append(f"facebook/opt-{size}")
                cmd.append("opt")
                cmd.append(str(shots))
                cmd.append(str(qbits))
                cmd = ' '.join(cmd)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")

    # Llama-2
    if run_llama_2:
        llama_2_dir = os.environ["LLAMA_PATH"]
        for size in ["7b", "13b"]:
            if size == "7b":
                tasks = ["rte", "piqa"]
            elif size == "13b":
                tasks = ["copa", "winogrande"]
            for task in tasks:
                for qbits in [8, 4, 2, 1]:
                    cmd = []
                    cmd.append("bash quant.sh")
                    cmd.append(task)
                    cmd.append(f"{llama_2_dir}/llama-2-{size}")
                    cmd.append("llama")
                    cmd.append(str(shots))
                    cmd.append(str(qbits))
                    cmd = ' '.join(cmd)
                    print(cmd)
                    os.system(cmd)
                    print("-------------------------------------------")
