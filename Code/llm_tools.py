
def describe_tools():
    tools  = []
    tools += _describe_validate_thermodynamic_consistency()
    tools += _describe_validate_stress_symmetry()
    tools += _describe_validate_objectivity()
    tools += _describe_validate_material_symmetry()
    tools += _describe_validate_ellipticity()
    tools += _describe_validate_growth_condition()
    tools += _describe_validate_energy_normalization()
    tools += _describe_validate_stress_normalization()
    tools += _describe_validate_non_negativity_of_strain_energy()

    return tools


def _describe_validate_thermodynamic_consistency():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_thermodynamic_consistency",
                "description": (
                    "Numerically validates the thermodynamic consistency of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_stress_symmetry():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_stress_symmetry",
                "description": (
                    "Numerically validates the symmetry of the Cauchy stress tensor derived from "
                    "the first Piola-Kirchhoff stress tensor prediction by the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_objectivity():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_objectivity",
                "description": (
                    "Numerically validates the objectivity (frame indifference) of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_material_symmetry():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_material_symmetry",
                "description": (
                    "Numerically validates the material symmetry of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_ellipticity():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_ellipticity",
                "description": (
                    "Numerically validates the ellipticity (Legendre–Hadamard condition) of the "
                    "most recently proposed constitutive model. Ellipticity is the property the "
                    "constitutive model should practically satisfy for material stability. In "
                    "construction, polyconvexity is the most effective route to ellipticity, "
                    "since polyconvexity implies ellipticity; this check therefore indicates "
                    "whether the efforts made to enforce polyconvexity during construction have "
                    "been sufficient. The model under test is supplied implicitly by the runtime; "
                    "this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_growth_condition():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_growth_condition",
                "description": (
                    "Numerically validates the growth condition of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_energy_normalization():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_energy_normalization",
                "description": (
                    "Numerically validates the energy normalization of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_stress_normalization():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_stress_normalization",
                "description": (
                    "Numerically validates the stress normalization of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _describe_validate_non_negativity_of_strain_energy():
    return [
        {
            "type": "function",
            "function": {
                "name": "validate_non_negativity_of_strain_energy",
                "description": (
                    "Numerically validates the non-negativity of the strain energy of the most recently "
                    "proposed constitutive model. The model under test is supplied implicitly by "
                    "the runtime; this tool takes no arguments. "
                    "Returns a JSON object with two keys: "
                    "'validation' (str, name or short description of the validation performed, and "
                    "'passed' (bool, whether the CANN satisfies that check)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]
