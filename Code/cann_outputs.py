import os
import json
import numpy             as np
import pandas            as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cann_inputs

def save_cann(config, refinement, creation_chat, creation, violating_creation, data, _extract_cann_prediction, best=False):
    if best:
        refinement_dir = os.path.join(config["output_dir"], "best")
    else:
        refinement_dir = os.path.join(config["output_dir"], f"refinement_{refinement}")
    os.makedirs(refinement_dir, exist_ok=True)

    path = os.path.join(refinement_dir, f"refinement_{refinement}.txt")
    with open(path, mode="w", encoding="utf-8") as file:
        pass

    path = os.path.join(refinement_dir, "creation_chat.txt")
    with open(path, mode="w", encoding="utf-8") as file:
        for message in creation_chat:
            file.write(f"{message['role']}: {message['content']}\n")

    path = os.path.join(refinement_dir, "cann_script.txt")
    with open(path, mode="w", encoding="utf-8") as file:
        file.write(creation["cann_script"])

    path = os.path.join(refinement_dir, "cann.weights.h5")
    creation["cann"].save_weights(path)

    if violating_creation is not None:
        path = os.path.join(refinement_dir, "exemplary_constraint_violating_cann_script.txt")
        with open(path, mode="w", encoding="utf-8") as file:
            file.write(violating_creation["cann_script"])

        path = os.path.join(refinement_dir, "exemplary_constraint_violating_cann.weights.h5")
        violating_creation["cann"].save_weights(path)

    path = os.path.join(refinement_dir, "metrics.txt")
    with open(path, mode="w", encoding="utf-8") as file:
        file.write("R2 Train\n" + "".join(f"{loading.replace('_', ' ')}: {loading_r2:.3f}\n" for loading, loading_r2 in creation["r2_train"].items()))
        file.write("\nR2 All\n" + "".join(f"{loading.replace('_', ' ')}: {loading_r2:.3f}\n" for loading, loading_r2 in creation["r2_all"].items()))

    if config["problem"] in ["synthetic_rubber", "experimental_rubber", "experimental_brain"]:
        _plot_isotropic_predictions(config, creation["cann"], data, creation["r2_all"], _extract_cann_prediction, refinement_dir)
    elif config["problem"] == "experimental_skin":
        _plot_anisotropic_predictions(config, creation["cann"], data, creation["r2_all"], _extract_cann_prediction, refinement_dir)
    else:
        raise NotImplementedError(f"Problem '{config['problem']}' is not implemented.")


def _plot_isotropic_predictions(config, cann, data, r2, _extract_cann_prediction, output_dir):
    colors = {
        "uni-x":   "#4169E1",
        "equi-x":  "#DC143C",
        "strip-x": "#FF8C00",
        "tens":    "#228B22",
        "comp":    "#C71585",
        "shear":   "#DAA520"
    }

    plt.figure()
    plt.title("CANN predictions")
    plt.plot([], [], linestyle="--", color="black", label="Prediction")
    plt.plot([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor="black", label="Training data")
    plt.plot([], [], linestyle="none", marker="s", markerfacecolor="black", markeredgecolor="black", label="Test data")

    for loading in data.keys():
        if loading == "all":
            continue

        x_idx = 1 if loading == "shear" else 0

        train_x_true = data[loading]["F"]["train"]
        train_y_true = data[loading]["P"]["train"]
        train_y_pred = cann.predict(train_x_true, verbose=0)["P"]
        train_x_true = train_x_true[:, 0, x_idx]
        train_y_true_extracted, train_y_pred_extracted = _extract_cann_prediction(config["problem"], train_y_true, train_y_pred)

        test_x_true = data[loading]["F"]["test"]
        test_y_true = data[loading]["P"]["test"]
        test_y_pred = cann.predict(test_x_true, verbose=0)["P"]
        test_x_true = test_x_true[:, 0, x_idx]
        test_y_true_extracted, test_y_pred_extracted = _extract_cann_prediction(config["problem"], test_y_true, test_y_pred)

        all_x_true = data[loading]["F"]["all"][:, 0, x_idx]
        if loading == "tens":
            all_x_true = np.insert(all_x_true, 0, data["comp"]["F"]["all"][-1, 0, x_idx], axis=0)

        min_x = np.min(all_x_true)
        max_x = np.max(all_x_true)

        extrapolation = 0.0
        range_x       = max_x - min_x
        if loading == "comp":
            extrapolated_min = min_x - extrapolation * range_x
            extrapolated_max = max_x
        else:
            extrapolated_min = min_x
            extrapolated_max = max_x + extrapolation * range_x

        plot_x_true = np.linspace(extrapolated_min, extrapolated_max, 2000)

        dl = cann_inputs.Dataloader(config)
        plot_F = dl._compute_F(loading, plot_x_true)

        plot_y_pred = cann.predict(plot_F, verbose=0)["P"]

        dummy_y_true = np.zeros((2000, 2), dtype=np.float32)
        if config["problem"] == "experimental_brain":
            dummy_y_true[:, 1] = dl._define_brain_loading_code(loading)

        _, plot_y_pred_extracted = _extract_cann_prediction(config["problem"], dummy_y_true, plot_y_pred)

        r2_loading = r2[f"on_{loading}_all_data"]
        
        plt.plot(plot_x_true, plot_y_pred_extracted, linestyle="--", color=colors[loading], label=f"{loading} (R2 on all data: {r2_loading:.3f})")
        plt.plot(train_x_true, train_y_true_extracted, linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=colors[loading])
        plt.plot(test_x_true, test_y_true_extracted, linestyle="none", marker="s", markerfacecolor=colors[loading], markeredgecolor=colors[loading])

        pd.DataFrame({"x": train_x_true, "y": train_y_true_extracted, "y_pred": train_y_pred_extracted}).to_csv(os.path.join(output_dir, f"{loading}_train_data.csv"), index=False)
        pd.DataFrame({"x":  test_x_true, "y":  test_y_true_extracted, "y_pred":  test_y_pred_extracted}).to_csv(os.path.join(output_dir, f"{loading}_test_data.csv"),  index=False)
        pd.DataFrame({"x":  plot_x_true,                              "y":       plot_y_pred_extracted}).to_csv(os.path.join(output_dir, f"{loading}_all_preds.csv"),  index=False)

    plt.grid()
    plt.xlabel("Stretch (λ)")
    plt.ylabel("Stress (P)")
    plt.legend()
    path = os.path.join(output_dir, "stress_predictions.png")
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_anisotropic_predictions(config, cann, data, r2, _extract_cann_prediction, output_dir):
    colors = {
        "strip-x": "#4169E1",
        "off-x":   "#DC143C",
        "biaxial": "#FF8C00",
        "off-y":   "#228B22",
        "strip-y": "#DAA520"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CANN predictions")
    for ax, direction in zip(axes, ["x", "y"]):
        ax.set_title(f"{direction}-direction")
        ax.set_xlabel("Stretch (λ)")
        ax.set_ylabel("Stress (σ)")
        ax.grid(True)
        ax.plot([], [], linestyle="--", color="black", label="Prediction")
        ax.plot([], [], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor="black", label="Training data")
        ax.plot([], [], linestyle="none", marker="s", markerfacecolor="black", markeredgecolor="black", label="Test data")

    dl = cann_inputs.Dataloader(config)

    for loading in data.keys():
        if loading == "all":
            continue

        # Measured train/test points with both stretches and the true/predicted stresses.
        splits = {}
        for split in ["train", "test"]:
            F     = data[loading]["F"][split]
            Sigma = data[loading]["Sigma"][split]
            P     = cann.predict(F, verbose=0)["P"]

            true_extracted, pred_extracted = _extract_cann_prediction(config["problem"], Sigma, P)
            true_extracted = np.asarray(true_extracted)
            pred_extracted = np.asarray(pred_extracted)

            df = pd.DataFrame({
                "lam_x":        F[:, 0, 0],
                "lam_y":        F[:, 1, 1],
                "sigma_x":      true_extracted[:, 0],
                "sigma_y":      true_extracted[:, 1],
                "sigma_x_pred": pred_extracted[:, 0],
                "sigma_y_pred": pred_extracted[:, 1]
            })
            df.to_csv(os.path.join(output_dir, f"{loading}_{split}_data.csv"), index=False)
            splits[split] = df

        # Dense prediction grid along the loading's stretch path. Each loading fixes the relation
        # between lam_x and lam_y, so densify the stretch with the larger range and interpolate the
        # other from the measured points (this yields lam = 1 exactly for the strip loadings and
        # follows the measured path for the off/biaxial loadings).
        all_lam_x = data[loading]["F"]["all"][:, 0, 0]
        all_lam_y = data[loading]["F"]["all"][:, 1, 1]
        if np.ptp(all_lam_x) >= np.ptp(all_lam_y):
            order      = np.argsort(all_lam_x)
            plot_lam_x = np.linspace(all_lam_x.min(), all_lam_x.max(), 2000)
            plot_lam_y = np.interp(plot_lam_x, all_lam_x[order], all_lam_y[order])
        else:
            order      = np.argsort(all_lam_y)
            plot_lam_y = np.linspace(all_lam_y.min(), all_lam_y.max(), 2000)
            plot_lam_x = np.interp(plot_lam_y, all_lam_y[order], all_lam_x[order])

        plot_lam = np.stack([plot_lam_x, plot_lam_y], axis=1).astype(np.float32)
        plot_F   = dl._compute_F(loading, plot_lam)
        plot_P   = cann.predict(plot_F, verbose=0)["P"]

        dummy_y_true       = np.zeros((2000, 4), dtype=np.float32)
        dummy_y_true[:, 2] = plot_lam_x
        dummy_y_true[:, 3] = plot_lam_y
        _, plot_pred_extracted = _extract_cann_prediction(config["problem"], dummy_y_true, plot_P)
        plot_pred_extracted    = np.asarray(plot_pred_extracted)

        pd.DataFrame({
            "lam_x":        plot_lam_x,
            "lam_y":        plot_lam_y,
            "sigma_x_pred": plot_pred_extracted[:, 0],
            "sigma_y_pred": plot_pred_extracted[:, 1]
        }).to_csv(os.path.join(output_dir, f"{loading}_all_preds.csv"), index=False)

        r2_x = r2[f"on_{loading}_x_all_data"]
        r2_y = r2[f"on_{loading}_y_all_data"]

        axes[0].plot(plot_lam_x, plot_pred_extracted[:, 0], linestyle="--", color=colors[loading], label=f"{loading} (R2 on all data: {r2_x:.3f})")
        axes[0].plot(splits["train"]["lam_x"], splits["train"]["sigma_x"], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=colors[loading])
        axes[0].plot(splits["test"]["lam_x"],  splits["test"]["sigma_x"],  linestyle="none", marker="s", markerfacecolor=colors[loading], markeredgecolor=colors[loading])
        axes[1].plot(plot_lam_y, plot_pred_extracted[:, 1], linestyle="--", color=colors[loading], label=f"{loading} (R2 on all data: {r2_y:.3f})")
        axes[1].plot(splits["train"]["lam_y"], splits["train"]["sigma_y"], linestyle="none", marker="o", markerfacecolor="none", markeredgecolor=colors[loading])
        axes[1].plot(splits["test"]["lam_y"],  splits["test"]["sigma_y"],  linestyle="none", marker="s", markerfacecolor=colors[loading], markeredgecolor=colors[loading])

    for ax in axes:
        ax.legend(fontsize="small")
    path = os.path.join(output_dir, "stress_predictions.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_run(config, refinement_chat, best_creation, data, _extract_cann_prediction, run_protocol):
    config_path = os.path.join(config["output_dir"], "config.json")
    with open(config_path, mode="w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    chat_path = os.path.join(config["output_dir"], "messages.txt")
    with open(chat_path, mode="w", encoding="utf-8") as file:
        for message in refinement_chat:
            file.write(f"{message['role']}: {message['content']}\n")

    protocol_path = os.path.join(config["output_dir"], "run_protocol.json")
    with open(protocol_path, mode="w", encoding="utf-8") as file:
        json.dump(run_protocol, file, indent=4)

    if best_creation["r2_avg"] > -np.inf:
        save_cann(config, best_creation["refinement"], best_creation["creation_chat"], best_creation, None, data, _extract_cann_prediction, True)
