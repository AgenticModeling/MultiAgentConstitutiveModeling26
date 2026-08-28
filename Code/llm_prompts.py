
def write_prompt(problem, role, agent):
    match problem:
        case "synthetic_rubber" | "experimental_rubber" | "experimental_brain":
            match role:
                case "system":
                    match agent:
                        case "creator":
                            return _write_isotropic_system_creator_prompt()
                        case "inspector":
                            return _write_isotropic_system_inspector_prompt()
                        case _:
                            raise ValueError(f"Agent '{agent}' is not implemented for role '{role}' for problem '{problem}'.")
                case "user":
                    match agent:
                        case "creator":
                            return _write_isotropic_user_creator_prompt()
                        case "inspector":
                            return _write_isotropic_user_inspector_prompt()
                        case _:
                            raise ValueError(f"Agent '{agent}' is not implemented for role '{role}' for problem '{problem}'.")
                case _:
                    raise ValueError(f"Role '{role}' is not implemented for problem '{problem}'.")
        case "experimental_skin":
            match role:
                case "system":
                    match agent:
                        case "creator":
                            return _write_transversely_isotropic_system_creator_prompt()
                        case "inspector":
                            return _write_transversely_isotropic_system_inspector_prompt()
                        case _:
                            raise ValueError(f"Agent '{agent}' is not implemented for role '{role}' for problem '{problem}'.")
                case "user":
                    match agent:
                        case "creator":
                            return _write_transversely_isotropic_user_creator_prompt()
                        case "inspector":
                            return _write_transversely_isotropic_user_inspector_prompt()
                        case _:
                            raise ValueError(f"Agent '{agent}' is not implemented for role '{role}' for problem '{problem}'.")
                case _:
                    raise ValueError(f"Role '{role}' is not implemented for problem '{problem}'.")
        case _:
            raise ValueError(f"Problem '{problem}' is not implemented.")


def _write_isotropic_system_creator_prompt():
    return '''You are an expert in constitutive modeling and artificial neural networks. Your task is to assist in generating a Python script that implements a Constitutive Artificial Neural Network (CANN) for a hyperelastic incompressible isotropic material. The script will compute the stress from the deformation gradient using the principles of continuum mechanics and neural networks. Follow the provided skeleton and guidelines carefully.'''


def _write_isotropic_system_inspector_prompt():
    return '''You are an expert in constitutive modeling and artificial neural networks. Your task is to verify the adherence to physical constraints of a generated Python script that implements a Constitutive Artificial Neural Network (CANN) for a hyperelastic incompressible isotropic material. The script will compute the stress from the deformation gradient using the principles of continuum mechanics and neural networks. Follow the provided guidelines carefully.'''


def _write_isotropic_user_creator_prompt():
    return '''
1.	Your task is to complete a Python script that implements a constitutive model predicting the strain energy density psi and the first Piola–Kirchhoff stress P from the deformation gradient F for a hyperelastic, incompressible, isotropic material using a Constitutive Artificial Neural Network (CANN).
2.	Your model must adhere to the following physical constraints:
2a.	Thermodynamic consistency: Use one or more neural-network blocks to predict the scalar strain-energy density psi. Obtain the isochoric part of the stress P by differentiating psi with respect to the deformation. Complement the isochoric part with the volumetric part of P based on the hydrostatic pressure p, which serves as a Lagrange multiplier enforcing incompressibility. Determine p from the boundary condition that the normal stress in the third spatial direction vanishes.
2b.	Symmetry of the stress tensor: Use the invariants of C as network inputs instead of the components of F or C directly.
2c.	Objectivity: Guaranteed by using the invariants of C as network inputs (see above).
2d.	Material symmetry: Guaranteed by using the invariants of C as network inputs (see above).
2e.	Polyconvexity: Preserve polyconvexity through the network. Constrain all weights to be non-negative. All activation functions must be convex and monotonically increasing. Additionally, they must be at least twice continuously differentiable. Suitable choices include (parametric) Softplus, Exponential, Squared Softplus, Gaussian-CDF integral, and Smooth ReLU variants.
2f.	Growth condition: Guaranteed automatically under incompressibility.
2g.	Energy normalization: Subtract psi evaluated at C = I from the network output so that psi(C = I) = 0.
2h.	Stress normalization: Guaranteed automatically via the Lagrange multiplier p under isotropic incompressibility.
2i.	Non-negativity of strain energy: Guaranteed automatically by the preceding constraints.
3.	Consider the following implementation hints:
3a.	The deformation gradient F serving as input will always be provided as a tf.Tensor of dtype tf.float32 and shape (batch_size,3,3). The model output must be a dictionary {"P": P, "Psi": psi}. The output psi is a tf.Tensor of dtype tf.float32 and shape (batch_size,1), while the output P is a tf.Tensor of dtype tf.float32 and shape (batch_size,3,3).
3b.	Automatic differentiation of psi can be cleanly accomplished in TensorFlow using tf.GradientTape. Open a tf.GradientTape() context, use tape.watch() to track the necessary tensors, and ensure all operations within the block form a differentiable chain from input to output. Then obtain the gradient using tape.gradient(). Note that tf.GradientTape only works with eager execution and cannot differentiate through Keras symbolic placeholders.
3c.	You are not allowed to use static methods in your implementation.
4.	Take your time to plan your implementation step by step.
5.	Please respond with a completion of the Python script skeleton provided below. Include the marker <BEGIN PYTHON SCRIPT> at the beginning of the code and <END PYTHON SCRIPT> at the end. Your response will be automatically parsed to extract the Python script between these markers. Please strictly follow the structure and function signatures provided in the skeleton.


<BEGIN PYTHON SCRIPT>
import tensorflow as tf


class CANN(tf.keras.Model):
    def __init__(self, **kwargs):
        """ 
        Args:
            kwargs: Additional keyword arguments for the model initialization.
        """

    def psi_from_F(self, F):
        """
        Pure forward computation of psi from F. No tf.GradientTape inside.

        Args:
            F (tf.Tensor): Input tensor containing the deformation gradient F. Shape: (batch_size,3,3).
        Returns:
            tf.Tensor: Strain energy density psi predicted from the deformation gradient F. Shape: (batch_size,1)
        """

    def call(self, F):
        """
        Uses method psi_from_F and tf.GradientTape to predict both psi and P from F.

        Args:
            F (tf.Tensor): Input tensor containing the deformation gradient F. Shape: (batch_size,3,3)
        Returns:
            dict: Dictionary with keys "P" and "Psi":
                  - "P": First Piola-Kirchhoff stress tensor. Shape: (batch_size,3,3).
                  - "Psi": Strain energy density. Shape: (batch_size,1).
        """


def build_cann_model():
    """ Builds a CANN model for predicting the strain energy density psi and the first Piola Kirchhoff stress P from the deformation gradient F for hyperelastic, incompressible, isotropic materials.
    Args:
        -- None --
    Returns:
        CANN: A CANN model instance (subclass of tf.keras.Model).
    """


<END PYTHON SCRIPT>
    
6. '''


def _write_isotropic_user_inspector_prompt():
    return '''
1.	Your task is to inspect the Python script below that implements a constitutive model predicting the strain energy density psi and the first Piola–Kirchhoff stress P from the deformation gradient F for a hyperelastic, incompressible, isotropic material using a Constitutive Artificial Neural Network (CANN).
2.	Your task is to verify the adherence of the model defined in the script to the following physical constraints:
2a.	Thermodynamic consistency: The stress must be computed as the derivative of a scalar energy potential, not predicted independently. Verify that the network has a single scalar output psi, and that the stress is obtained via automatic differentiation of that output with respect to the deformation. The incompressibility constraint is enforced through a Lagrange multiplier pressure p to be determined from the boundary condition that the normal stress in the third spatial direction vanishes.
2b.	Symmetry of the stress tensor: The stress tensor must be symmetric to satisfy conservation of angular momentum. This is guaranteed if the network input is formulated only in terms of invariants of C. Under incompressibility, I3 = 1 is constant. Verify that the network takes exactly two inputs: I1 = tr(C) and I2 = tr(cof(C)), and that no other quantities enter the network. Note that under the incompressibility constraint det(F) = 1, it holds that I1 >= 3 and I2 >= 3 for all admissible deformations.
2c.	Objectivity: The energy must be unchanged when a rigid rotation is superimposed on the current configuration. This is ensured by the same criterion as point 2.
2d.	Material symmetry: The energy must be unchanged under any rotation of the reference configuration. This is ensured by the same criterion as point 2.
2e.	Polyconvexity: Polyconvexity implies material stability and prevents unphysical behavior in boundary value problems. Verify three aspects: (a) all weights in the network are constrained to be non-negative, (b) all activation functions are convex and monotonically increasing over the admissible input domain, and (c) all activation functions are at least twice continuously differentiable. If any weight is negative or any activation function violates convexity, monotonicity, or smoothness, polyconvexity is not guaranteed.
2f.	Growth condition: The energy must tend to infinity as volume tends to zero or infinity. The incompressibility constraint automatically guarantees this. 
2g.	Energy normalization: The energy must be zero in the undeformed configuration. Verify that the network output includes a subtraction of psi evaluated at C = I, so that the final energy is zero in the undeformed state.
2h.	Stress normalization: The material must be stress-free in the undeformed configuration. The combination of isotropy and the incompressibility constraint automatically guarantees this.
2i.	Non-negativity of strain energy: The energy must be non-negative for all admissible deformation states. The combination of polyconvexity, energy normalization, and stress normalization automatically guarantees this.
3.  Your task is to inspect the provided constitutive model for compliance with the specified constraints yourself whenever possible. Use a corresponding numerical validation tool only if you are genuinely uncertain whether a specific constraint is fulfilled.
4.	Below is the CANN you are to inspect. Provide your feedback in the following JSON form: {"Thermodynamic consistency fulfilled": false, "Thermodynamic consistency explanation": "", "Symmetry of the stress tensor fulfilled": false, "Symmetry of the stress tensor explanation": "", "Objectivity fulfilled": false, "Objectivity explanation": "", "Material symmetry fulfilled": false, "Material symmetry explanation": "", "Polyconvexity fulfilled": false, "Polyconvexity explanation": "", "Growth condition fulfilled": false, "Growth condition explanation": "", "Energy normalization fulfilled": false, "Energy normalization explanation": "", "Stress normalization fulfilled": false, "Stress normalization explanation": "", "Non-negativity of strain energy fulfilled": false, "Non-negativity of strain energy explanation": ""}. For each constraint fulfillment, provide exactly one of the two possible values: True if the condition is fulfilled, and False if the condition is violated. In addition, provide your explanation on why this condition is fulfilled or violated for each constraint as a brief text. Please strictly follow this form.

'''


def _write_transversely_isotropic_system_creator_prompt():
    return '''You are an expert in constitutive modeling and artificial neural networks. Your task is to assist in generating a Python script that implements a Constitutive Artificial Neural Network (CANN) for a hyperelastic incompressible transversely isotropic material. The script will compute the stress from the deformation gradient using the principles of continuum mechanics and neural networks. Follow the provided skeleton and guidelines carefully.'''


def _write_transversely_isotropic_system_inspector_prompt():
    return '''You are an expert in constitutive modeling and artificial neural networks. Your task is to verify the adherence to physical constraints of a generated Python script that implements a Constitutive Artificial Neural Network (CANN) for a hyperelastic incompressible transversely isotropic material. The script will compute the stress from the deformation gradient using the principles of continuum mechanics and neural networks. Follow the provided guidelines carefully.'''


def _write_transversely_isotropic_user_creator_prompt():
    return '''
1.	Your task is to complete a Python script that implements a constitutive model predicting the strain energy density psi and the first Piola–Kirchhoff stress P from the deformation gradient F for a hyperelastic, incompressible, transversely isotropic material using a Constitutive Artificial Neural Network (CANN).
2.	Your model must adhere to the following physical constraints:
2a.	Thermodynamic consistency: Use one or more neural-network blocks to predict the scalar strain-energy density psi. Obtain the isochoric part of the stress P by differentiating psi with respect to the deformation. Complement the isochoric part with the volumetric part of P based on the hydrostatic pressure p, which serves as a Lagrange multiplier enforcing incompressibility. Determine p from the boundary condition that the normal stress in the third spatial direction vanishes.
2b.	Symmetry of the stress tensor: Use the invariants of C, together with the pseudo-invariants formed from C and the fiber structure tensor (see material symmetry below), as network inputs instead of the components of F or C directly.
2c.	Objectivity: Guaranteed by using these invariants and pseudo-invariants as network inputs (see above).
2d.	Material symmetry: The material is transversely isotropic, exhibiting a single preferred fiber direction. Represent the fiber by a direction vector n that is assumed to lie in the xy-plane and is parameterized by a trainable angle alpha as n = [cos(alpha), sin(alpha), 0], so that the fiber direction is learned from data. Use exactly one fiber family: a single trainable angle alpha and a single structure tensor N shared by all fiber terms. Build the structure tensor N = n ⊗ n from the fiber direction and incorporate the preferred direction through the pseudo-invariants I4 = C : N and I5 = cof(C) : N, where : denotes the double contraction of tensors and cof(C) = det(C) C^(-T) is the cofactor of C. Compute the cofactor explicitly, either inversion-free via the Cayley-Hamilton identity cof(C) = C^2 - I1*C + I2*I or as det(C) * C^(-1) with the determinant factor retained. Do not simplify cof(C) to C^(-1), even though det(C) = 1 holds for all admissible deformations: stresses and elasticity tensors are derivatives with respect to all nine components of F, which leave the incompressibility manifold, and only the full cofactor form remains polyconvex there. I4 measures the squared stretch along the fiber and I5 the squared areal stretch of the plane normal to the fiber; both are polyconvex, in contrast to the classical alternative C^2 : N, which must not be used. Together with the isotropic invariants I1 and I2, the pseudo-invariants I4 and I5 serve as the inputs to the network. Note that I4 and I5 equal 1 in the reference configuration and, unlike I1 and I2 (which satisfy I1 >= 3 and I2 >= 3 under incompressibility), can fall below their reference value under admissible deformations; center any input shifts at 1, not at 3.
2e.	Polyconvexity: Preserve polyconvexity through the network. Constrain all weights to be non-negative. All activation functions must be convex and monotonically increasing. Additionally, they must be at least twice continuously differentiable. Suitable choices include (parametric) Softplus, Exponential, Squared Softplus, Gaussian-CDF integral, and Smooth ReLU variants.
2f.	Growth condition: Guaranteed automatically under incompressibility.
2g.	Energy normalization: Subtract psi evaluated at C = I from the network output so that psi(C = I) = 0.
2h.	Stress normalization: Under transverse isotropy the Lagrange multiplier p does not by itself remove the fiber-induced stress at the reference configuration. Enforce a stress-free reference explicitly by adding the correction term psi_stress = c4 * (I4 - 1) + c5 * (I5 - 1) to psi. Exploit that at C = I the derivatives dI4/dC = N and dI5/dC = I - N carry the structure tensor with opposite signs: with x = (dpsi/dI4 - dpsi/dI5) evaluated at C = I, the non-negative choice c4 = relu(-x), c5 = relu(x) cancels the fiber-aligned part of the reference stress. Because c4 and c5 are non-negative and I4 and I5 are polyconvex, this correction preserves polyconvexity, whereas subtracting invariant terms with negative coefficients would destroy it. Recompute c4 and c5 from the current network weights in every forward pass. The remaining isotropic part of the reference stress is then completed to P = 0 by the pressure p.
2i.	Non-negativity of strain energy: Unlike in the isotropic case, non-negativity is not guaranteed automatically under transverse isotropy, because I4 and I5 can fall below their reference value 1, where monotonically increasing contributions drop below their value at the reference configuration and become negative after energy normalization. Ensure by design that psi >= 0 for all admissible deformations. One sufficient strategy is to let the fiber contributions act on (I4 - 1) and (I5 - 1) through convex, non-decreasing, twice continuously differentiable activations that vanish together with their first derivative for non-positive arguments (Smooth ReLU variants); this keeps every fiber term non-negative and additionally makes the stress correction from 2h vanish identically.
3.	Consider the following implementation hints:
3a.	The deformation gradient F serving as input will always be provided as a tf.Tensor of dtype tf.float32 and shape (batch_size,3,3). The model output must be a dictionary {"P": P, "Psi": psi}. The output psi is a tf.Tensor of dtype tf.float32 and shape (batch_size,1), while the output P is a tf.Tensor of dtype tf.float32 and shape (batch_size,3,3).
3b.	Automatic differentiation of psi can be cleanly accomplished in TensorFlow using tf.GradientTape. Open a tf.GradientTape() context, use tape.watch() to track the necessary tensors, and ensure all operations within the block form a differentiable chain from input to output. Then obtain the gradient using tape.gradient(). Note that tf.GradientTape only works with eager execution and cannot differentiate through Keras symbolic placeholders.
3c.	You are not allowed to use static methods in your implementation.
4.	Take your time to plan your implementation step by step.
5.	Please respond with a completion of the Python script skeleton provided below. Include the marker <BEGIN PYTHON SCRIPT> at the beginning of the code and <END PYTHON SCRIPT> at the end. Your response will be automatically parsed to extract the Python script between these markers. Please strictly follow the structure and function signatures provided in the skeleton.


<BEGIN PYTHON SCRIPT>
import tensorflow as tf


class CANN(tf.keras.Model):
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Additional keyword arguments for the model initialization.
        """

    def psi_from_F(self, F):
        """
        Pure forward computation of psi from F. No tf.GradientTape inside.

        Args:
            F (tf.Tensor): Input tensor containing the deformation gradient F. Shape: (batch_size,3,3).
        Returns:
            tf.Tensor: Strain energy density psi predicted from the deformation gradient F. Shape: (batch_size,1)
        """

    def call(self, F):
        """
        Uses method psi_from_F and tf.GradientTape to predict both psi and P from F.

        Args:
            F (tf.Tensor): Input tensor containing the deformation gradient F. Shape: (batch_size,3,3)
        Returns:
            dict: Dictionary with keys "P" and "Psi":
                  - "P": First Piola-Kirchhoff stress tensor. Shape: (batch_size,3,3).
                  - "Psi": Strain energy density. Shape: (batch_size,1).
        """


def build_cann_model():
    """ Builds a CANN model for predicting the strain energy density psi and the first Piola Kirchhoff stress P from the deformation gradient F for hyperelastic, incompressible, transversely isotropic materials.
    Args:
        -- None --
    Returns:
        CANN: A CANN model instance (subclass of tf.keras.Model).
    """


<END PYTHON SCRIPT>

6. '''


def _write_transversely_isotropic_user_inspector_prompt():
    return '''
1.	Your task is to inspect the Python script below that implements a constitutive model predicting the strain energy density psi and the first Piola–Kirchhoff stress P from the deformation gradient F for a hyperelastic, incompressible, transversely isotropic material using a Constitutive Artificial Neural Network (CANN).
2.	Your task is to verify the adherence of the model defined in the script to the following physical constraints:
2a.	Thermodynamic consistency: The stress must be computed as the derivative of a scalar energy potential, not predicted independently. Verify that the network has a single scalar output psi, and that the stress is obtained via automatic differentiation of that output with respect to the deformation. The incompressibility constraint is enforced through a Lagrange multiplier pressure p to be determined from the boundary condition that the normal stress in the third spatial direction vanishes.
2b.	Symmetry of the stress tensor: The stress tensor must be symmetric to satisfy conservation of angular momentum. This is guaranteed if the network input is formulated only in terms of invariants of C and the pseudo-invariants formed from C and the structure tensor N. Under incompressibility, I3 = 1 is constant. Verify that the network takes exactly four inputs: I1 = tr(C), I2 = tr(cof(C)), I4 = C : N, and I5 = cof(C) : N with the cofactor computed in full, e.g. via the Cayley-Hamilton identity cof(C) = C^2 - I1*C + I2*I or as det(C) * C^(-1) with the determinant factor retained, and that no other quantities enter the network. Note that under the incompressibility constraint det(F) = 1, it holds that I1 >= 3 and I2 >= 3 for all admissible deformations, whereas the pseudo-invariants I4 and I5 are positive with reference value 1 and can fall below 1 under admissible deformations.
2c.	Objectivity: The energy must be unchanged when a rigid rotation is superimposed on the current configuration. This is ensured by the same criterion as point 2.
2d.	Material symmetry: The material is transversely isotropic, so the energy must be unchanged only under those transformations of the reference configuration that preserve the fiber direction: rotations about the fiber axis and reflections that map the fiber onto itself up to sign. Verify that the preferred direction is represented by a fiber vector n = [cos(alpha), sin(alpha), 0] lying in the xy-plane with a trainable angle alpha, that the structure tensor N = n ⊗ n is formed from it, and that the anisotropy enters the network exclusively through the pseudo-invariants I4 = C : N and I5 = cof(C) : N. The classical alternative I5 = C^2 : N is not polyconvex and must be flagged as a violation. Likewise, implementing I5 as C^(-1) : N with the determinant factor of the cofactor dropped destroys polyconvexity away from the incompressibility manifold, where stresses and elasticity tensors are evaluated by differentiation, and must equally be flagged.
2e.	Polyconvexity: Polyconvexity implies material stability and prevents unphysical behavior in boundary value problems. Verify three aspects: (a) all weights in the network are constrained to be non-negative, (b) all activation functions are convex and monotonically increasing over the admissible input domain, and (c) all activation functions are at least twice continuously differentiable. If any weight is negative or any activation function violates convexity, monotonicity, or smoothness, polyconvexity is not guaranteed.
2f.	Growth condition: The energy must tend to infinity as volume tends to zero or infinity. The incompressibility constraint automatically guarantees this.
2g.	Energy normalization: The energy must be zero in the undeformed configuration. Verify that the network output includes a subtraction of psi evaluated at C = I, so that the final energy is zero in the undeformed state.
2h.	Stress normalization: The material must be stress-free in the undeformed configuration. For transverse isotropy this is not guaranteed by incompressibility alone; verify that the model explicitly removes the fiber-induced reference stress so that P(F = I) = 0, e.g. through a correction term linear in I4 and I5 with non-negative coefficients chosen such that the fiber-aligned reference stress cancels (possible because dI4/dC = N and dI5/dC = I - N at C = I carry the structure tensor with opposite signs). A correction that subtracts invariant terms with negative coefficients destroys polyconvexity and must be flagged.
2i.	Non-negativity of strain energy: The energy must be non-negative for all admissible deformation states. In contrast to the isotropic case, the combination of polyconvexity, energy normalization, and stress normalization does not automatically guarantee this under transverse isotropy, because the pseudo-invariants I4 and I5 can fall below their reference value 1, where monotonically increasing contributions drop below their value at the reference configuration. Verify that the model keeps psi non-negative nonetheless, e.g. through fiber contributions that vanish together with their first derivative at and below the reference (Smooth ReLU variants acting on I4 - 1 and I5 - 1).
3.  Your task is to inspect the provided constitutive model for compliance with the specified constraints yourself whenever possible. Use a corresponding numerical validation tool only if you are genuinely uncertain whether a specific constraint is fulfilled.
4.	Below is the CANN you are to inspect. Provide your feedback in the following JSON form: {"Thermodynamic consistency fulfilled": false, "Thermodynamic consistency explanation": "", "Symmetry of the stress tensor fulfilled": false, "Symmetry of the stress tensor explanation": "", "Objectivity fulfilled": false, "Objectivity explanation": "", "Material symmetry fulfilled": false, "Material symmetry explanation": "", "Polyconvexity fulfilled": false, "Polyconvexity explanation": "", "Growth condition fulfilled": false, "Growth condition explanation": "", "Energy normalization fulfilled": false, "Energy normalization explanation": "", "Stress normalization fulfilled": false, "Stress normalization explanation": "", "Non-negativity of strain energy fulfilled": false, "Non-negativity of strain energy explanation": ""}. For each constraint fulfillment, provide exactly one of the two possible values: True if the condition is fulfilled, and False if the condition is violated. In addition, provide your explanation on why this condition is fulfilled or violated for each constraint as a brief text. Please strictly follow this form.

'''
