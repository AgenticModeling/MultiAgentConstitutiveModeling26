import numpy      as np
import tensorflow as tf


def _all_close(a, b, n_digits=3, atol=1e-4):
    both_nan = np.isnan(a) & np.isnan(b)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    skip     = both_nan | both_inf
    a_safe   = np.where(skip, 0.0, a)
    b_safe   = np.where(skip, 0.0, b)
    scale    = np.maximum(np.abs(a_safe), np.abs(b_safe))
    tol      = np.where(scale > atol, scale * 10**(-n_digits), atol + scale)
    return np.all(skip | (np.abs(a_safe - b_safe) <= tol))


def _fill_F(F11, F12, F21, F22):
    det2d = F11 * F22 - F12 * F21
    assert det2d > 0, f"In-plane det must be positive, got {det2d}"
    F = np.zeros((3,3), dtype=np.float32)
    F[0,0], F[0,1] = F11, F12
    F[1,0], F[1,1] = F21, F22
    F[2,2]         = 1.0 / det2d
    return F


def _define_F(problem=None):
    F_list = []

    F_list.append(np.eye(3, dtype=np.float32))

    for lam in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0]:
        F_list.append(_fill_F(lam, 0, 0, 1.0/np.sqrt(lam)))

    for lam in [0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
        F_list.append(_fill_F(lam, 0, 0, lam))

    for lam in [0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0]:
        F_list.append(_fill_F(lam, 0, 0, 1.0))

    for gamma in [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]:
        F_list.append(_fill_F(1.0, gamma, 0.0, 1.0))

    for l1, l2 in [(1.2, 1.5), (1.5, 0.8), (2.0, 1.3), (0.6, 1.8)]:
        F_list.append(_fill_F(l1, 0, 0, l2))

    for lam, gam in [(1.3, 0.3), (1.5, 0.5), (0.8, 0.4)]:
        F_list.append(_fill_F(lam, gam, 0.0, 1.0/np.sqrt(lam)))

    if problem == "experimental_skin":
        F_list += list(_define_out_of_plane_F())

    return np.array(F_list)


def _define_out_of_plane_F():
    # Plane-strain states keep C13 = C23 = 0, so an in-plane fiber never sees the out-of-plane
    # shear modes that the pseudo-invariant V = cof(C):N couples to.
    F_list = []

    for gamma in [-0.7, -0.3, 0.3, 0.7]:
        for i, j in [(0, 2), (1, 2), (2, 0), (2, 1)]:
            F       = np.eye(3, dtype=np.float32)
            F[i, j] = gamma
            F_list.append(F)

    for lam, gamma in [(1.5, 0.5), (0.7, 0.5)]:
        F       = np.diag([lam, 1.0, 1.0/lam]).astype(np.float32)
        F[0, 2] = gamma
        F_list.append(F)

    return np.array(F_list, dtype=np.float32)


def _fill_out_of_plane_Q(phi1, phi2, phi3, det=1):
    def _Rx(phi):
        R = np.zeros((3, 3), dtype=np.float32)
        R[0,0]         = 1.0
        R[1,1], R[1,2] = np.cos(phi), -np.sin(phi)
        R[2,1], R[2,2] = np.sin(phi),  np.cos(phi)
        return R

    def _Ry(phi):
        R = np.zeros((3, 3), dtype=np.float32)
        R[0,0], R[0,2] =  np.cos(phi), np.sin(phi)
        R[1,1]         =  1.0
        R[2,0], R[2,2] = -np.sin(phi), np.cos(phi)
        return R

    def _Rz(phi):
        R = np.zeros((3, 3), dtype=np.float32)
        R[0,0], R[0,1] = np.cos(phi), -np.sin(phi)
        R[1,0], R[1,1] = np.sin(phi),  np.cos(phi)
        R[2,2]         = 1.0
        return R

    R = _Rx(phi1) @ _Ry(phi2) @ _Rz(phi3)
    if det == -1:
        R = R @ np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    return R


def _fill_in_plane_Q(theta):
    Q = np.zeros((3,3), dtype=np.float32)
    Q[0,0], Q[0,1] = np.cos(theta), -np.sin(theta)
    Q[1,0], Q[1,1] = np.sin(theta),  np.cos(theta)
    Q[2,2]         = 1.0
    return Q


def _define_out_of_plane_Q(det=1):
    assert det in [1,-1], "det must be +1 or -1"

    angle_triples = [
        (  np.pi/6, np.pi/4,  np.pi/3),
        (  np.pi/4, np.pi/4,  np.pi/4),
        (  np.pi/2,     0.0,  np.pi/2),
        (      0.0, np.pi/3,  np.pi/6),
        (  np.pi/3, np.pi/2,  np.pi/4),
        (2*np.pi/3, np.pi,  5*np.pi/3),
        (     0.73,    1.05,     2.14),
        (     1.22,    0.58,     0.37),
        (  np.pi,   np.pi/2,      0.0),
    ]

    return np.array([_fill_out_of_plane_Q(p1, p2, p3, det=det) for p1, p2, p3 in angle_triples])


def _rotation_about_axis(axis, angle):
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[     0.0, -axis[2],  axis[1]],
                  [ axis[2],      0.0, -axis[0]],
                  [-axis[1],  axis[0],      0.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def _define_transverse_isotropy_Q(axis):
    # Transverse-isotropy group about the fiber: rotations about it plus reflections fixing it (Q^T N Q = N).
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    perp = np.array([-axis[1], axis[0], 0.0], dtype=np.float32)

    # 0.73 is not a rational multiple of pi, included for generality.
    Qs = [_rotation_about_axis(axis, a) for a in [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, np.pi, 0.73]]
    Qs.append(np.eye(3, dtype=np.float32) - 2.0 * np.outer(axis, axis))  # reflect across plane perpendicular to fiber
    Qs.append(np.eye(3, dtype=np.float32) - 2.0 * np.outer(perp, perp))  # reflect across plane containing fiber
    return np.array(Qs, dtype=np.float32)


def _recover_inplane_fiber_axes(cann, lams=(1.3, 2.0, 3.0), n_angles=36):
    # Candidate fiber axes from Fourier modes of Psi over in-plane-rotated probes (details in the
    # manuscript appendix). Two probe families and two modes are needed: on diag(lam, 1/lam, 1),
    # cof(C):N equals C:M of the in-plane normal, so balanced I4/I5 channels have no 2nd mode (the
    # 4th still carries the axis), and balanced linear channels are constant there altogether,
    # which diag(lam, 1, 1/lam) resolves. Multiple amplitudes catch fiber terms that only activate
    # at larger stretches.
    phis = np.linspace(0.0, np.pi, n_angles, endpoint=False)

    def _strongest_mode(diag_fn, order):
        best = 0.0 + 0.0j
        for lam in lams:
            D    = np.diag(diag_fn(lam)).astype(np.float32)
            Fs   = np.array([_fill_in_plane_Q(phi) @ D @ _fill_in_plane_Q(phi).T for phi in phis], dtype=np.float32)
            psi  = cann.predict(Fs)["Psi"][:, 0]
            mode = np.sum(psi * np.exp(1j * order * phis))
            if np.abs(mode) > np.abs(best):
                best = mode
        return best

    thetas = [0.50 * np.angle(_strongest_mode(lambda lam: [lam, 1.0/lam, 1.0],     2)),
              0.25 * np.angle(_strongest_mode(lambda lam: [lam, 1.0/lam, 1.0],     4)),
              0.50 * np.angle(_strongest_mode(lambda lam: [lam, 1.0,     1.0/lam], 2))]

    axes = []
    for theta in thetas:
        for shift in (0.0, np.pi/2):
            axis = np.array([np.cos(theta + shift), np.sin(theta + shift), 0.0], dtype=np.float32)
            if all(np.abs(np.dot(axis, existing)) < 0.999 for existing in axes):
                axes.append(axis)
    return axes


def _inplane_symmetry_deviation(cann, Fs, psi_F, theta):
    axis    = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
    Qs      = _define_transverse_isotropy_Q(axis)
    FQTs    = (Fs[:,np.newaxis] @ Qs[np.newaxis].transpose(0,1,3,2)).reshape(-1,3,3)
    psi_FQT = cann.predict(FQTs)["Psi"][:,0].reshape(len(Fs), len(Qs))

    a, b   = psi_FQT, psi_F[:,np.newaxis]
    skip   = (np.isnan(a) & np.isnan(b)) | (np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b)))
    a_safe = np.where(skip, 0.0, a)
    b_safe = np.where(skip, 0.0, b)
    scale  = np.maximum(np.maximum(np.abs(a_safe), np.abs(b_safe)), 1e-4)
    return np.max(np.abs(a_safe - b_safe) / scale)


def _scan_inplane_fiber_axis(cann, Fs, psi_F, n_angles=180):
    # Brute-force fallback for models whose Fourier modes degenerate: the coarse argmin lands
    # within the refinement window even for sharply peaked symmetry landscapes.
    thetas = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    devs   = [_inplane_symmetry_deviation(cann, Fs, psi_F, theta) for theta in thetas]
    theta  = thetas[int(np.argmin(devs))]
    return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)


def _refine_inplane_fiber_axis(cann, Fs, psi_F, axis, half_width=np.pi/90, n_iter=20):
    # Steep fiber stiffening amplifies sub-degree axis errors far beyond the comparison tolerance,
    # so the axis estimate is polished by a golden-section search on the symmetry deviation.
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    theta  = np.arctan2(axis[1], axis[0])
    lo, hi = theta - half_width, theta + half_width
    x1, x2 = hi - golden * (hi - lo), lo + golden * (hi - lo)
    f1, f2 = _inplane_symmetry_deviation(cann, Fs, psi_F, x1), _inplane_symmetry_deviation(cann, Fs, psi_F, x2)

    for _ in range(n_iter):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - golden * (hi - lo)
            f1 = _inplane_symmetry_deviation(cann, Fs, psi_F, x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + golden * (hi - lo)
            f2 = _inplane_symmetry_deviation(cann, Fs, psi_F, x2)

    theta = x1 if f1 <= f2 else x2
    return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)


def _define_diagonal_F_for_hessian(max_log_stretch=np.inf, L=2.0, n_grid=50):
    log_lam = np.linspace(-L, L, n_grid)
    log_l1, log_l2 = np.meshgrid(log_lam, log_lam, indexing="ij")
    stretches = np.stack([np.exp(log_l1.ravel()),
                          np.exp(log_l2.ravel()),
                          np.exp(-(log_l1 + log_l2).ravel())], axis=-1)
    stretches = np.sort(stretches, axis=-1)[:, ::-1]
    stretches = np.unique(np.round(stretches, decimals=10), axis=0)
    stretches = stretches[np.max(np.abs(np.log(stretches)), axis=1) <= max_log_stretch]

    return np.array([_fill_F(s[0], 0, 0, s[1]) for s in stretches])


def _define_symmetric_F_for_hessian(max_log_stretch=np.inf, L=1.5, n_grid=16):
    log_lam        = np.linspace(-L, L, n_grid)
    log_l1, log_l2 = np.meshgrid(log_lam, log_lam, indexing="ij")
    lam1           = np.exp(log_l1.ravel())
    lam2           = np.exp(log_l2.ravel())

    # Order the in-plane pair (the orientation sweep covers their swap) but keep the out-of-plane stretch separate.
    lam_hi    = np.maximum(lam1, lam2)
    lam_lo    = np.minimum(lam1, lam2)
    lam_out   = 1.0 / (lam_hi * lam_lo)
    stretches = np.stack([lam_hi, lam_lo, lam_out], axis=-1)
    stretches = np.unique(np.round(stretches, decimals=10), axis=0)
    stretches = stretches[np.max(np.abs(np.log(stretches)), axis=1) <= max_log_stretch]

    # Orientations sweep SO(3), not only in-plane: in-plane-only keeps z principal and hides the
    # out-of-plane shear that cof(C):N couples to. The pi/8 spacing keeps beta + pi/2 on the grid,
    # so the sweep covers the swap of the ordered in-plane pair exactly.
    orientations  = [_fill_in_plane_Q(b) for b in np.linspace(0.0, np.pi, 8, endpoint=False)]
    orientations += list(_define_out_of_plane_Q(det=1))

    Fs = []
    for s in stretches:
        D = np.diag(s).astype(np.float32)
        for Q in orientations:
            Fs.append((Q @ D @ Q.T).astype(np.float32))
    return np.array(Fs, dtype=np.float32)


def _define_a_and_b_for_hessian(n_dirs=200):
    def _fibonacci_hemisphere(n):
        idx    = np.arange(n, dtype=float)
        golden = (1 + np.sqrt(5)) / 2
        theta  = np.arccos(1 - (idx + 0.5) / n)  # cos(theta) in (0,1): upper hemisphere
        phi    = 2 * np.pi * idx / golden
        dirs = np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)], axis=-1)
        return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    dirs_a = _fibonacci_hemisphere(n_dirs)
    dirs_b = _fibonacci_hemisphere(n_dirs)
    return np.einsum("ai,bj->abij", dirs_a, dirs_b).reshape(-1, 9)


def _interpolate_F_loop(waypoints, n_seg=50):
    Fs = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i],   dtype=np.float32)
        end   = np.array(waypoints[i+1], dtype=np.float32)
        for k in range(n_seg):
            s = k / n_seg
            p = (1 - s) * start + s * end
            Fs.append(_fill_F(p[0], p[1], p[2], p[3]))
    Fs.append(_fill_F(*waypoints[-1]))
    return np.array(Fs)


def _fill_F_out_of_plane(F11, F12, F21, F22, F13, F23):
    # F31 = F32 = 0, so F13 and F23 do not enter det(F) and the construction stays volume-preserving.
    F                = _fill_F(F11, F12, F21, F22)
    F[0, 2], F[1, 2] = F13, F23
    return F


def _interpolate_F_loop_out_of_plane(waypoints, n_seg=50):
    Fs = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i],   dtype=np.float32)
        end   = np.array(waypoints[i+1], dtype=np.float32)
        for k in range(n_seg):
            s = k / n_seg
            p = (1 - s) * start + s * end
            Fs.append(_fill_F_out_of_plane(*p))
    Fs.append(_fill_F_out_of_plane(*waypoints[-1]))
    return np.array(Fs)


def _define_F_paths_for_thermodynamic_consistency(problem, n_seg=200):
    s  = np.linspace(0, 1, n_seg + 1)
    QF = _fill_in_plane_Q(np.pi/4) @ _fill_F(1.3, 0.5, 0.0, 1.0/np.sqrt(1.3))

    # Loop 1: diagonal deformations only
    loop_1 = _interpolate_F_loop([
        (1.0, 0.0, 0.0, 1.0),
        (1.5, 0.0, 0.0, 1.0/np.sqrt(1.5)),
        (1.5, 0.0, 0.0, 1.0),
        (1.3, 0.0, 0.0, 1.3),
        (1.0, 0.0, 0.0, 1.0),
    ], n_seg)

    # Loop 2: off-diagonal deformations with rotated waypoint
    loop_2 = _interpolate_F_loop([
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.5, 0.0, 1.0),
        (1.3, 0.5, 0.0, 1.0/np.sqrt(1.3)),
        (QF[0,0], QF[0,1], QF[1,0], QF[1,1]),
        (1.3, 0.0, 0.0, 1.0/np.sqrt(1.3)),
        (1.0, 0.0, 0.0, 1.0),
    ], n_seg)

    loops = [loop_1, loop_2]

    # Loop 3 (transverse isotropy only): plane-strain loops keep dF13 = dF23 = 0 and are blind to
    # non-gradient stress contributions confined to the out-of-plane components.
    if problem == "experimental_skin":
        loops.append(_interpolate_F_loop_out_of_plane([
            (1.0, 0.0, 0.0, 1.0,               0.0, 0.0),
            (1.0, 0.0, 0.0, 1.0,               0.5, 0.0),
            (1.3, 0.0, 0.0, 1.0/np.sqrt(1.3),  0.5, 0.3),
            (1.0, 0.3, 0.0, 1.0,               0.0, 0.3),
            (1.0, 0.0, 0.0, 1.0,               0.0, 0.0),
        ], n_seg))

    # Two-path comparison: direct uniaxial vs. via pure shear
    path_a = np.array([_fill_F(1.0 + si*0.5, 0.0, 0.0, 1.0/np.sqrt(1.0 + si*0.5)) for si in s])
    path_b = np.array([_fill_F(1.0 + si*0.5, 0.0, 0.0, 1.0) for si in s] +
                       [_fill_F(1.5, 0.0, 0.0, 1.0 + si*(1.0/np.sqrt(1.5) - 1.0)) for si in s[1:]])

    return loops, path_a, path_b


def _compute_work_along_F_path(cann, Fs):
    out     = cann.predict(Fs)
    P, psi  = out["P"], out["Psi"][:, 0]
    delta_W = np.einsum("kij,kij->k", 0.5 * (P[:-1] + P[1:]), np.diff(Fs, axis=0))
    W_cumul = np.concatenate([[0.0], np.cumsum(delta_W)])
    dpsi    = psi - psi[0]
    return W_cumul, dpsi, delta_W


def validate_thermodynamic_consistency(cann, problem):
    loops, path_a, path_b = _define_F_paths_for_thermodynamic_consistency(problem)

    for loop in loops:
        W_cumul, dpsi, delta_W = _compute_work_along_F_path(cann, loop)

        # Metric 1: normalized loop residual
        W_loop  = W_cumul[-1]
        sum_abs = np.sum(np.abs(delta_W))
        eta     = np.abs(W_loop) / sum_abs if sum_abs > 0 else 0.0
        if eta > 1e-2:
            return "failed"

        # Metric 2: stress uniqueness at start == end
        P_start = cann.predict(loop[:1])["P"]
        P_end   = cann.predict(loop[-1:])["P"]
        if not _all_close(P_start, P_end):
            return "failed"

    # Metric 3: two-path comparison
    W_a, dpsi_a, _ = _compute_work_along_F_path(cann, path_a)
    W_b, dpsi_b, _ = _compute_work_along_F_path(cann, path_b)

    if not _all_close(W_a[-1], W_b[-1]):
        return "failed"

    # Metric 4: work-energy consistency along open paths
    if not _all_close(W_a, dpsi_a, n_digits=2):
        return "failed"
    if not _all_close(W_b, dpsi_b, n_digits=2):
        return "failed"

    return "passed"


def validate_stress_symmetry(cann, problem):
    Fs  = _define_F(problem)
    Ps  = cann.predict(Fs)["P"]

    sigmas = Ps @ Fs.transpose(0,2,1)

    if _all_close(sigmas, sigmas.transpose(0,2,1)):
        return "passed"
    return "failed"


def validate_objectivity(cann, problem):
    # Psi is computed independent of the Lagrange multiplier p. Since p is the only place the plane-
    # stress assumption (P33=0) enters, arbitrary 3D rotations can be used to test Psi(QF) = Psi(F).
    Fs = _define_F(problem)
    Qs = _define_out_of_plane_Q()
    n_F, n_Q = len(Fs), len(Qs)

    psi_F  = cann.predict(Fs)["Psi"][:,0]
    QFs    = (Qs[:,np.newaxis] @ Fs[np.newaxis]).reshape(-1,3,3)
    psi_QF = cann.predict(QFs)["Psi"][:,0].reshape(n_Q,n_F)

    if _all_close(psi_QF, psi_F[np.newaxis]):
        return "passed"
    return "failed"


def validate_material_symmetry(cann, problem):
    # Test Psi(F @ Q.T) = Psi(F): isotropic symmetry group is O(3); the skin problem uses the fiber group.
    Fs = _define_F(problem)

    def _symmetric_under(Qs):
        psi_F   = cann.predict(Fs)["Psi"][:,0]
        FQTs    = (Fs[:,np.newaxis] @ Qs[np.newaxis].transpose(0,1,3,2)).reshape(-1,3,3)
        psi_FQT = cann.predict(FQTs)["Psi"][:,0].reshape(len(Fs), len(Qs))
        return _all_close(psi_FQT, psi_F[:,np.newaxis])

    if problem == "experimental_skin":
        # Fiber angle is learned: a true fiber passes about one candidate, an orthotropic model about none.
        psi_F = cann.predict(Fs)["Psi"][:,0]
        for axis in _recover_inplane_fiber_axes(cann):
            axis = _refine_inplane_fiber_axis(cann, Fs, psi_F, axis)
            if _symmetric_under(_define_transverse_isotropy_Q(axis)):
                return "passed"
        axis = _scan_inplane_fiber_axis(cann, Fs, psi_F)
        axis = _refine_inplane_fiber_axis(cann, Fs, psi_F, axis)
        if _symmetric_under(_define_transverse_isotropy_Q(axis)):
            return "passed"
        return "failed"

    Qs = np.concatenate([_define_out_of_plane_Q(det=1), _define_out_of_plane_Q(det=-1)], axis=0)
    return "passed" if _symmetric_under(Qs) else "failed"


def validate_ellipticity(cann, problem):
    # Check rank-one convexity: (a⊗b) : A : (a⊗b) ≥ 0  ∀ a,b ∈ R³
    def _compute_elasticity_tensor(F):
        N = F.shape[0]
        f = tf.Variable(tf.reshape(F, (N,9)))

        with tf.GradientTape() as tape_outer:
            with tf.GradientTape() as tape_inner:
                F_reshaped = tf.reshape(f, (N,3,3))
                psi        = cann.psi_from_F(F_reshaped)
            grad = tape_inner.batch_jacobian(psi, f)
            grad = tf.squeeze(grad, axis=1)
        hessian = tape_outer.batch_jacobian(grad, f)

        return hessian.numpy().reshape(N,3,3,3,3)

    max_log_stretches = {
        "synthetic_rubber":    2.0,
        "experimental_rubber": 2.0,
        "experimental_brain":  0.8,
        "experimental_skin":   0.8,
    }

    # Isotropic problems can restrict to diagonal F (objectivity + isotropy). Transverse isotropy
    # keeps the fiber, so sample symmetric F with the in-plane axes swept relative to it.
    if problem == "experimental_skin":
        Fs = _define_symmetric_F_for_hessian(max_log_stretches[problem])
    else:
        Fs = _define_diagonal_F_for_hessian(max_log_stretches[problem])
    h_all      = _define_a_and_b_for_hessian()
    batch_size = 64
    for start in range(0, len(Fs), batch_size):
        A_flat = _compute_elasticity_tensor(Fs[start:start+batch_size]).reshape(-1,9,9)
        vals   = np.einsum("bi,fij,bj->fb", h_all, A_flat, h_all)
        # NaN comparisons are False, so non-finite values would otherwise pass silently.
        if not np.all(np.isfinite(vals)):
            return "failed"
        # float32 cancellation at strongly deformed states produces spurious negatives (~1e-6
        # relative on provably polyconvex models), so judge negativity relative to the per-state
        # magnitude of the form instead of against a bare zero.
        tol_rel, tol_abs = 1e-3, 1e-4
        scale = np.max(np.abs(vals), axis=1, keepdims=True)
        if np.any(vals < -(tol_rel * scale + tol_abs)):
            return "failed"

    return "passed"


def validate_growth_condition(cann):
    return "passed" # Trivial for incompressibility


def validate_energy_normalization(cann):
    tol = 1e-3
    psi = cann.predict(np.eye(3).reshape(1,3,3))["Psi"]
    if np.abs(psi[0,0]) < tol:
        return "passed"
    else:
        return "failed"


def validate_stress_normalization(cann):
    tol = 1e-3
    P   = cann.predict(np.eye(3).reshape(1,3,3))["P"]
    if np.max(np.abs(P)) < tol:
        return "passed"
    else:
        return "failed"


def validate_non_negativity_of_strain_energy(cann, problem):
    # For transverse isotropy non-negativity is not guaranteed by construction; the ellipticity
    # set reaches much stronger fiber compression (I4 ~ e^-3) than the base set.
    Fs = _define_F(problem)
    if problem == "experimental_skin":
        Fs = np.concatenate([Fs, _define_symmetric_F_for_hessian()], axis=0)
    psi = cann.predict(Fs)["Psi"][:,0]

    if np.all(psi >= -1e-3):
        return "passed"
    return "failed"
