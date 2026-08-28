import re
import json
import random
import numpy           as np
import traceback       as tb
import tensorflow      as tf
import sklearn.metrics as metrics

import llm_tools
import llm_prompts
import cann_outputs
import cann_validations


def run(config, llm, data):
    run_protocol       = {}
    cann_handling      = _define_cann_handling()
    cann_handling_hint = f"Your CANN implementation will be integrated with the following functions. Please ensure that your implementation adheres to the expected function signatures, input/output types, and tensor shapes.\n\n{cann_handling}"
    ns_cann_handling   = {}
    exec(cann_handling, ns_cann_handling)

    best_creation           = {}
    best_creation["r2_avg"] = -np.inf

    refinement_chat = []
    _extend_chat(refinement_chat, "system", text=llm_prompts.write_prompt(config["problem"], "system", "creator"))
    _extend_chat(refinement_chat, "user",   text=llm_prompts.write_prompt(config["problem"],   "user", "creator") + cann_handling_hint)

    for refinement in range(config["n_refinement_attempts"]):
        _extend_protocol(run_protocol, "start of new refinement round")
        print("\033[38;5;208mStarting a new refinement!\033[0m")

        creation_chat        = refinement_chat.copy()
        violating_creations  = []
        n_constraint_attempt = 0
        abort_refinement     = False

        while True:
            creation = create_cann(config, llm, creation_chat, ns_cann_handling, data, run_protocol)
            if creation is None:
                abort_refinement = True
                break

            inspection = inspect_cann(config, llm, creation, run_protocol)
            if inspection is None:
                abort_refinement = True
                break

            if not inspection["constraint_violations"]:
                break

            violating_creations += [creation]

            if n_constraint_attempt >= config["n_constraint_attempts"]:
                print(f"\033[91mCANN continues to violate constraints after {config['n_constraint_attempts']} attempts!\033[0m")
                _extend_protocol(run_protocol, "physically correct cann creation aborted", f"CANN continues to violate constraints after {config['n_constraint_attempts']} attempts!")
                abort_refinement = True
                break

            n_constraint_attempt += 1

            print(f"\033[91mCANN violates constraints!\033[0m")
            _extend_chat(creation_chat, "user", text=inspection["constraint_violations_msg"])

        if abort_refinement:
            break

        print("\033[92mPhysically correct CANN creation succeeded!\n" + "".join(f"R2 {loading.replace('_', ' ')}: {             loading_r2:.3f}\n" for loading, loading_r2 in creation["r2_train"].items()) + "\033[0m")
        _extend_protocol(run_protocol, "physically correct cann creation succeeded", "".join(f"R2 {loading.replace('_', ' ')}: {loading_r2:.3f}\n" for loading, loading_r2 in creation["r2_train"].items()).strip())
        _extend_chat(refinement_chat, "assistant", text=creation["cann_script"])
        _extend_chat(creation_chat,   "user",      text=f"The CANN you implemented with your previous Python script satisfies all required constraints - thank you! On the training data, it produces the following predictions:\n{creation['predictions']}Do you think you can further improve this performance by refining your implementation? Please provide a new and complete Python script that builds on your earlier version and continues to follow all originally stated constraints and hints.")
        _extend_chat(refinement_chat, "user",      text=f"The CANN you implemented with your previous Python script satisfies all required constraints - thank you! On the training data, it produces the following predictions:\n{creation['predictions']}Do you think you can further improve this performance by refining your implementation? Please provide a new and complete Python script that builds on your earlier version and continues to follow all originally stated constraints and hints.")

        violating_creation = random.choice(violating_creations) if violating_creations else None
        cann_outputs.save_cann(config, refinement, creation_chat, creation, violating_creation, data, ns_cann_handling["_extract_cann_prediction"])

        if creation["r2_avg"] > best_creation["r2_avg"]:
            best_creation["refinement"]    = refinement
            best_creation["creation_chat"] = creation_chat
            best_creation["cann_script"]   = creation["cann_script"]
            best_creation["cann"]          = creation["cann"]
            best_creation["r2_train"]      = creation["r2_train"]
            best_creation["r2_all"]        = creation["r2_all"]
            best_creation["r2_avg"]        = creation["r2_avg"]

    cann_outputs.save_run(config, refinement_chat, best_creation, data, ns_cann_handling["_extract_cann_prediction"], run_protocol)


def create_cann(config, llm, creation_chat, ns_cann_handling, data, run_protocol):
    creation_attempt = 0
    error_chat       = creation_chat.copy()

    while creation_attempt < config["n_creation_attempts"]:
        creation = _create_cann(config, llm, error_chat, ns_cann_handling, data, run_protocol)
        creation_attempt += 1

        if not creation["error"]:
            print(f"\033[92mCANN creation succeeded!\033[0m")
            _extend_chat(creation_chat, "assistant", text=error_chat[-1]["content"])
            return creation

        print(f"\033[91mCANN creation failed with error {creation['error_msg']}\033[0m")
        _extend_chat(error_chat, "user", text=creation["error_msg"])

    print(f"\033[91mCANN creation failed after {config['n_creation_attempts']} attempts!\033[0m")
    _extend_protocol(run_protocol, "cann creation aborted", f"CANN creation failed after {config['n_creation_attempts']} attempts!")
    return None


def _create_cann(config, llm, creation_chat, cann_handling, data, run_protocol):
    try:
        response, _ = llm.respond(creation_chat)
        _extend_chat(creation_chat, "assistant", text=response)
    except RuntimeError as e:
        raise RuntimeError("Failed to generate CANN from LLM") from e

    try:
        cann_script = cann_handling["_extract_cann_script"](response)
        cann        = cann_handling["_execute_cann_script"](cann_script)
    except Exception as e:
        msg = f"Failed to extract and execute CANN script from LLM response: {str(e)}"
        _extend_protocol(run_protocol, "syntactical error in cann creation", msg)
        return {"error": True, "error_msg": msg + f"\n\n{tb.format_exc()}"}

    try:
        cann_handling["_compile_cann"](  config, cann, data)
        cann_handling["_fit_cann"](      config, cann, data)
        cann_handling["_load_best_cann"](config, cann)
    except Exception as e:
        msg = f"Failed to compile, fit, and load best CANN: {str(e)}"
        _extend_protocol(run_protocol, "syntactical error in created cann interaction with static code", msg)
        return {"error": True, "error_msg": msg + f"\n\n{tb.format_exc()}"}

    try:
        _test_cann(config, cann_script, cann_handling)
    except Exception as e:
        msg = f"Failed to test CANN: {str(e)}"
        _extend_protocol(run_protocol, "syntactical error in created cann interaction with static code", msg)
        return {"error": True, "error_msg": msg + f"\n\n{tb.format_exc()}"}

    try:
        r2_train, r2_all, predictions = _evaluate_cann(config, cann, cann_handling, data)
        r2_avg                        = np.mean(list(r2_train.values()))
    except Exception as e:
        msg = f"Failed to evaluate CANN: {str(e)}"
        _extend_protocol(run_protocol, "syntactical error in created cann interaction with static code", msg)
        return {"error": True, "error_msg": msg + f"\n\n{tb.format_exc()}"}

    _extend_protocol(run_protocol, "successful cann creation", "".join(f"R2 {loading.replace('_', ' ')}: {loading_r2:.3f}\n" for loading, loading_r2 in r2_train.items()).strip())

    return {"error": False, "cann_script": cann_script, "cann": cann, "r2_train": r2_train, "r2_all": r2_all, "r2_avg": r2_avg, "predictions": predictions}


def inspect_cann(config, llm, creation, run_protocol):
    inspection_chat = []
    _extend_chat(inspection_chat, "system", text=llm_prompts.write_prompt(config["problem"], "system", "inspector"))
    _extend_chat(inspection_chat, "user",   text=llm_prompts.write_prompt(config["problem"],   "user", "inspector") + creation["cann_script"])

    inspection_attempt = 0

    while inspection_attempt < config["n_inspection_attempts"]:
        inspection = _inspect_cann(creation["cann"], config["problem"], llm, inspection_chat, run_protocol)
        inspection_attempt += 1

        if not inspection["error"]:
            print(f"\033[92mCANN inspection succeeded!\033[0m")
            return inspection

        print(f"\033[91mCANN inspection failed with error {inspection['error_msg']}\033[0m")
        _extend_chat(inspection_chat, "user", text=inspection["error_msg"])

    print(f"\033[91mCANN inspection failed after {config['n_inspection_attempts']} attempts!\033[0m")
    _extend_protocol(run_protocol, "cann inspection aborted", f"CANN inspection failed after {config['n_inspection_attempts']} attempts!")
    return None


def _inspect_cann(cann, problem, llm, inspection_chat, run_protocol):
    tools          = llm_tools.describe_tools()
    max_tool_calls = 9
    
    try:
        for _ in range(max_tool_calls + 1):
            response, tool_calls = llm.respond(inspection_chat, tools)
            if tool_calls:
                _extend_chat(inspection_chat, "assistant", tool_calls=tool_calls)
                inspection_chat += _handle_inspector_tool_calls(tool_calls, cann, problem, run_protocol)
            else:
                _extend_chat(inspection_chat, "assistant", text=response)
                break
        else:
            msg = f"Inspector exceeded {max_tool_calls} tool calls without returning a final response."
            _extend_protocol(run_protocol, "syntactical error in cann inspection", msg)
            return {"error": True, "error_msg": msg}
    except RuntimeError as e:
        msg = f"Failed to inspect CANN with LLM: {str(e)}"
        _extend_protocol(run_protocol, "syntactical error in cann inspection", msg)
        return {"error": True, "error_msg": msg + f"\n\n{tb.format_exc()}"}

    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if not matches:
        msg = "Failed to detect JSON object."
        _extend_protocol(run_protocol, "syntactical error in cann inspection", msg)
        return {"error": True, "error_msg": msg}

    try:
        inspection = json.loads(matches[-1])
    except json.JSONDecodeError as e:
        msg = f"Failed to parse JSON from response: {str(e)}"
        _extend_protocol(run_protocol, "syntactical error in cann inspection", msg)
        return {"error": True, "error_msg": msg + f"\n\n{tb.format_exc()}"}

    violated_constraints = {
        key.replace(" fulfilled", ""): inspection[f"{key.replace(' fulfilled', '')} explanation"]
        for key, value in inspection.items()
        if key.endswith(" fulfilled") and value is False
    }

    _extend_protocol(run_protocol, "successful cann inspection")

    if violated_constraints:
        msg = "\nConstraint violations detected:\n"
        for constraint, reasoning in violated_constraints.items():
            msg += f"\n- {constraint}:\n  {reasoning}"
        msg_out = msg + "\n\nPlease fix these violations in your implementation and try again."
        _extend_protocol(run_protocol, "constraint violation", msg.strip())
        return {"error": False, "constraint_violations": True, "constraint_violations_msg": msg_out}

    _extend_protocol(run_protocol, "constraint fulfillment")

    return {"error": False, "constraint_violations": False}


def _handle_inspector_tool_calls(tool_calls, cann, problem, run_protocol):
    results = []

    for tool_call in tool_calls:
        match tool_call.function.name:
            case "validate_thermodynamic_consistency":
                result = cann_validations.validate_thermodynamic_consistency(cann, problem)
            case "validate_stress_symmetry":
                result = cann_validations.validate_stress_symmetry(cann, problem)
            case "validate_objectivity":
                result = cann_validations.validate_objectivity(cann, problem)
            case "validate_material_symmetry":
                result = cann_validations.validate_material_symmetry(cann, problem)
            case "validate_ellipticity":
                result = cann_validations.validate_ellipticity(cann, problem)
            case "validate_growth_condition":
                result = cann_validations.validate_growth_condition(cann)
            case "validate_energy_normalization":
                result = cann_validations.validate_energy_normalization(cann)
            case "validate_stress_normalization":
                result = cann_validations.validate_stress_normalization(cann)
            case "validate_non_negativity_of_strain_energy":
                result = cann_validations.validate_non_negativity_of_strain_energy(cann, problem)
            case _:
                raise RuntimeError(f"Received unsupported tool call with name '{tool_call.function.name}' from LLM inspector!")
        
        _extend_protocol(run_protocol, "inspector tool call", msg=f"Inspector successfully called {tool_call.function.name} with result: {result}")

        results += [{
            "role":         "tool",
            "tool_call_id": tool_call.id,
            "content":      json.dumps({"validation": tool_call.function.name.removeprefix("validate_"), "passed": result=="passed"}),
        }]

    return results


def _define_cann_handling():
    return '''
import os
import re
import numpy           as np
import tensorflow      as tf


def _extract_cann_script(assistant_response):
    pattern_begin      = r"<\\s*BEGIN\\s+PYTHON\\s+SCRIPT\\s*>"
    pattern_end        = r"<\\s*END\\s+PYTHON\\s+SCRIPT\\s*>"
    num_patterns_begin = len(re.findall(pattern_begin, assistant_response))
    num_patterns_end   = len(re.findall(pattern_end,   assistant_response))

    if num_patterns_begin != 1 or \
       num_patterns_end   != 1:
        raise ValueError("Malformed response: exactly one <BEGIN PYTHON SCRIPT> and one <END PYTHON SCRIPT> marker expected!")

    pattern     = r"<\\s*BEGIN\\s+PYTHON\\s+SCRIPT\\s*>(.*?)<\\s*END\\s+PYTHON\\s+SCRIPT\\s*>"
    cann_script = re.findall(pattern, assistant_response, re.DOTALL)[0]
    return cann_script


def _execute_cann_script(cann_script):
    namespace_cann = {}
    exec(cann_script, namespace_cann)
    cann = namespace_cann["build_cann_model"]()
    return cann


def _compile_cann(config, cann, data):
    # For skin the loss compares two normal stresses that differ in magnitude. Weight each
    # direction by the inverse of its mean-square magnitude (computed once on the training data)
    # and normalize the weights to mean 1, so both axes contribute equally to the loss and the
    # weighting reduces to the unweighted loss when the two axes already share the same scale.
    loss_weights = None
    if config["problem"] == "experimental_skin":
        sigma        = data["all"]["Sigma"]["train"][:, :2]
        scale        = np.mean(np.square(sigma), axis=0)
        loss_weights = 1.0 / (scale + 1e-12)
        loss_weights = (loss_weights / np.mean(loss_weights)).astype(np.float32)

    cann.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss      = {"P": lambda y_true, y_pred: _compute_cann_loss(config["problem"], y_true, y_pred, loss_weights), "Psi": None},
    )


def _extract_cann_prediction(problem, y_true, y_pred):
    if problem in ["synthetic_rubber", "experimental_rubber"]:
        return y_true[:,0], y_pred[:,0,0]

    elif problem == "experimental_brain":
        def extract_sample(sample):
            y_true_sample, y_pred_sample = sample
            return tf.cond(
                tf.equal(y_true_sample[1], 2),
                lambda: (y_true_sample[0], y_pred_sample[0,1]),
                lambda: (y_true_sample[0], y_pred_sample[0,0])
            )
        extracted = tf.map_fn(fn=extract_sample, elems=(y_true, y_pred), dtype=(tf.float32, tf.float32))
        return extracted[0], extracted[1]

    elif problem == "experimental_skin":
        # y_true columns: [sigma_x, sigma_y, lam_x, lam_y]; convert the predicted first
        # Piola-Kirchhoff stress P to the Cauchy normal stresses via sigma = P F^T (J=1, F diagonal).
        y_true_extracted = tf.stack([y_true[:,0],                y_true[:,1]],                axis=1)
        y_pred_extracted = tf.stack([y_pred[:,0,0]*y_true[:,2],  y_pred[:,1,1]*y_true[:,3]],  axis=1)
        return y_true_extracted, y_pred_extracted

    else:
        raise NotImplementedError(f"Problem '{problem}' is not implemented.")


def _compute_cann_loss(problem, y_true, y_pred, weights=None):
    y_true_extracted, y_pred_extracted = _extract_cann_prediction(problem, y_true, y_pred)
    squared_error = tf.square(y_true_extracted - y_pred_extracted)
    if weights is not None:
        squared_error = squared_error * weights
    return tf.reduce_mean(squared_error)


def _fit_cann(config, cann, data):
    if config["problem"] in ["synthetic_rubber", "experimental_rubber", "experimental_brain"]:
        data_x = data["all"]["F"]["train"]
        data_y = data["all"]["P"]["train"]
    elif config["problem"] == "experimental_skin":
        data_x = data["all"]["F"]["train"]
        data_y = data["all"]["Sigma"]["train"]
    else:
        raise NotImplementedError(f"Problem '{config['problem']}' is not implemented.")
    
    cann.fit(
        x                = data_x,
        y                = data_y,
        verbose          = 0,
        batch_size       = config["cann_training_batch_size"],
        epochs           = config["cann_training_epochs"],
        validation_split = 0.0,
        shuffle          = True,
        callbacks = [tf.keras.callbacks.ModelCheckpoint(
            filepath          = os.path.join(config["temp_dir"], "cann.weights.h5"),
            monitor           = "loss",
            verbose           = 0,
            save_best_only    = True,
            mode              = "min",
            save_weights_only = True
        )]
    )


def _load_best_cann(config, cann):
    cann.load_weights(os.path.join(config["temp_dir"], "cann.weights.h5"))
'''


def _test_cann(config, cann_script, cann_handling):
    F_dummy = np.array([
        np.eye(3, dtype=np.float32),
        np.diag([1.5, 1.0 / np.sqrt(1.5), 1.0 / np.sqrt(1.5)]).astype(np.float32),
        np.diag([2.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]).astype(np.float32),
    ])
    N_dummy = F_dummy.shape[0]

    cann = _test_cann_loading(config, cann_script, cann_handling, F_dummy)
    _test_cann_method_predict(   cann, F_dummy, N_dummy)
    _test_cann_method_psi_from_F(cann, F_dummy, N_dummy)


def _test_cann_loading(config, cann_script, cann_handling, F_dummy):
    cann = cann_handling["_execute_cann_script"](cann_script)
    _    = cann.predict(F_dummy, verbose=0)
    _    = cann_handling["_load_best_cann"](config, cann)
    return cann


def _test_cann_method_predict(cann, F, N):
    output = cann.predict(F, verbose=0)

    assert isinstance(output, dict), "CANN predict output must be a dictionary!"
    assert "P"   in output,          "CANN predict output must contain key 'P'!"
    assert "Psi" in output,          "CANN predict output must contain key 'Psi'!"
    assert output["P"].shape   == (N, 3, 3), f"Expected P shape ({N}, 3, 3), got {output['P'].shape}!"
    assert output["Psi"].shape == (N, 1),    f"Expected Psi shape ({N}, 1), got {output['Psi'].shape}!"


def _test_cann_method_psi_from_F(cann, F, N):
    F_tensor = tf.constant(F)
    psi      = cann.psi_from_F(F_tensor)
    assert psi.shape == (N, 1), f"Expected psi shape ({N}, 1), got {psi.shape}!"

    f = tf.Variable(tf.reshape(F, (N, 9)))

    with tf.GradientTape() as tape_outer:
        with tf.GradientTape() as tape_inner:
            F_reshaped = tf.reshape(f, (N, 3, 3))
            psi        = cann.psi_from_F(F_reshaped)
        grad = tape_inner.batch_jacobian(psi, f)
        grad = tf.squeeze(grad, axis=1)
    hessian = tape_outer.batch_jacobian(grad, f)

    assert grad.shape    == (N, 9),    f"Expected grad shape ({N}, 9), got {grad.shape}!"
    assert hessian.shape == (N, 9, 9), f"Expected hessian shape ({N}, 9, 9), got {hessian.shape}!"
    assert not np.any(np.isnan(grad.numpy())),    "Gradient of psi_from_F contains NaN values!"
    assert not np.any(np.isnan(hessian.numpy())), "Hessian of psi_from_F contains NaN values!"


def _evaluate_cann(config, cann, cann_handling, data):
    if config["problem"] not in ["synthetic_rubber", "experimental_rubber", "experimental_brain", "experimental_skin"]:
        raise NotImplementedError(f"Problem '{config['problem']}' is not implemented.")

    axes = ["x", "y"] if config["problem"] == "experimental_skin" else [None]

    r2_train                = {}
    r2_all                  = {}
    predictions_description = ""
    for loading in data.keys():
        if loading == "all":
            continue

        train_vals, all_vals = _compute_r2(config, cann, cann_handling, data[loading])
        for axis, train_val, all_val in zip(axes, train_vals, all_vals):
            suffix = "" if axis is None else f"_{axis}"
            r2_train[f"on_{loading}{suffix}_train_data"] = float(train_val)
            r2_all[  f"on_{loading}{suffix}_all_data"]   = float(all_val)
        predictions_description += _describe_predictions(config, cann, cann_handling, loading, data[loading])

    return r2_train, r2_all, predictions_description


def _compute_r2(config, cann, cann_handling, data):
    stress = "Sigma" if config["problem"] == "experimental_skin" else "P"

    y_true_train =              data[stress]["train"]
    y_pred_train = cann.predict(data["F"]["train"], verbose=0)["P"]
    y_true_train_extracted, y_pred_train_extracted = cann_handling["_extract_cann_prediction"](config["problem"], y_true_train, y_pred_train)
    r2_train     = metrics.r2_score(y_true_train_extracted, y_pred_train_extracted, multioutput="raw_values")

    y_true_all =              data[stress]["all"]
    y_pred_all = cann.predict(data["F"]["all"], verbose=0)["P"]
    y_true_all_extracted, y_pred_all_extracted = cann_handling["_extract_cann_prediction"](config["problem"], y_true_all, y_pred_all)
    r2_all     = metrics.r2_score(y_true_all_extracted, y_pred_all_extracted, multioutput="raw_values")

    return r2_train, r2_all


def _describe_predictions(config, cann, cann_handling, loading, data):
    is_skin = config["problem"] == "experimental_skin"

    description = f"Predictions for {loading}:\n"

    train_F    = data["F"]["train"]
    train_true = data["Sigma"]["train"] if is_skin else data["P"]["train"]
    pred_P     = cann.predict(train_F, verbose=0)["P"]

    true_extracted, pred_extracted = cann_handling["_extract_cann_prediction"](config["problem"], train_true, pred_P)
    true_extracted = np.asarray(true_extracted)
    pred_extracted = np.asarray(pred_extracted)

    if is_skin:
        description += "Stretch X,Stretch Y,True Sigma X,True Sigma Y,Predicted Sigma X,Predicted Sigma Y\n"
        for i in range(len(train_F)):
            description += f"{train_F[i, 0, 0]:.4f},{train_F[i, 1, 1]:.4f},{true_extracted[i, 0]:.4f},{true_extracted[i, 1]:.4f},{pred_extracted[i, 0]:.4f},{pred_extracted[i, 1]:.4f}\n"
    else:
        description += "Strain,True Stress,Predicted Stress\n"
        for i in range(len(train_F)):
            description += f"{train_F[i, 0, 0]:.4f},{true_extracted[i]:.4f},{pred_extracted[i]:.4f}\n"

    if is_skin:
        r2  = metrics.r2_score(           true_extracted, pred_extracted, multioutput="raw_values")
        mse = metrics.mean_squared_error( true_extracted, pred_extracted, multioutput="raw_values")
        description += f"R2 Score X: {r2[0]:.4f}\n"
        description += f"R2 Score Y: {r2[1]:.4f}\n"
        description += f"MSE Score X: {mse[0]:.4f}\n"
        description += f"MSE Score Y: {mse[1]:.4f}\n\n"
    else:
        r2_score  = metrics.r2_score(          true_extracted, pred_extracted)
        mse_score = metrics.mean_squared_error(true_extracted, pred_extracted)
        description += f"R2 Score: {r2_score:.4f}\n"
        description += f"MSE Score: {mse_score:.4f}\n\n"
    return description


def _extend_chat(chat, role, text=None, tool_calls=None):
    if text is not None:
        chat += [{"role": role, "content": text}]

    if tool_calls is not None:
        chat += [{"role": role, "tool_calls": [tc.model_dump() for tc in tool_calls]}]


def _extend_protocol(protocol, entry_type, msg=None):
    entry = {"type": entry_type}
    if msg is not None:
        entry["msg"] = msg
    protocol[len(protocol)] = entry
