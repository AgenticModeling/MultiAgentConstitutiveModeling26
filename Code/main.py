import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import datetime
import concurrent.futures

import cann_inputs
import cann_generation
import llm


def main():
    main_single_problem("experimental_rubber")


def main_single_problem(problem):
    runs    = 15
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for run in range(runs):
            future = executor.submit(main_single_thread, run, problem)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            future.result()


def main_single_thread(run, problem):
    print("\033[38;5;208m\nStarting a new run!\033[0m")

    config = _set_config(problem)
    config = _create_output_and_temp_dir(config, run)

    loader = cann_inputs.Dataloader(config)
    loader.load()
    data   = loader.get_data()

    llm_instance = llm.LLM(config["llm"])
    llm_instance.set_up()

    cann_generation.run(config, llm_instance, data)


def _set_config(problem):
    return {
        "problem":                  problem, # "synthetic_rubber", "experimental_rubber", "experimental_brain", "experimental_skin"
        "output_dir":               None,
        "temp_dir":                 None,
        "llm":                      "anthropic/claude-sonnet-4.6",
        "cann_training_batch_size": 4,
        "cann_training_epochs":     4000, # 8000 for skin
        "n_creation_attempts":      4,
        "n_inspection_attempts":    4,
        "n_constraint_attempts":    4,
        "n_refinement_attempts":    3,
    }


def _create_output_and_temp_dir(config, run):
    timestamp            = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config["output_dir"] = os.path.join("..", "output", f"timestamp_{timestamp}_run_{run}")
    os.mkdir(config["output_dir"])
    config["temp_dir"]   = os.path.join(config["output_dir"], "temp")
    os.mkdir(config["temp_dir"])
    return config


if __name__ == "__main__":
    main()
