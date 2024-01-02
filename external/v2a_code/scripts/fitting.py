import torch
import trimesh
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F

SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]

# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)


def save_trimesh(verts, faces, fname):
    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(fname)
 

def edge_loss(faces, vertices, target_vertices):
    edges1 = (faces[:, 0], faces[:, 1])
    edges2 = (faces[:, 1], faces[:, 2])
    edges3 = (faces[:, 2], faces[:, 0])
    target_e1 = target_vertices[:, edges1[0]] - target_vertices[:, edges1[1]]
    target_e2 = target_vertices[:, edges2[0]] - target_vertices[:, edges2[1]]
    target_e3 = target_vertices[:, edges3[0]] - target_vertices[:, edges3[1]]
    target_e = torch.cat((target_e1, target_e2, target_e3), dim=1)

    e1 = vertices[:, edges1[0]] - vertices[:, edges1[1]]
    e2 = vertices[:, edges2[0]] - vertices[:, edges2[1]]
    e3 = vertices[:, edges3[0]] - vertices[:, edges3[1]]
    e = torch.cat((e1, e2, e3), dim=1)
    dist = F.mse_loss(e, target_e, reduction="none")
    return dist.mean()


def sem_loss(vertices, target_vertices, sealed_vertices_sem_idx, sem_idx):
    if len(sem_idx) == 0:
        return 0
    # sem_idx assume sealed MANO left/right hands
    # however, it is fine as long as we don't the palm region as the sealed vertex is there.
    # this works for left/right hands
    # tip_sem_idx = [12, 11, 4, 5, 6]  # thumb, index, .., pinky tips
    vidx = np.concatenate([sealed_vertices_sem_idx[sem_idx] for sem_idx in sem_idx])
    vertices_sem = vertices[:, vidx]
    target_vertices_sem = target_vertices[:, vidx]
    dist = F.mse_loss(vertices_sem, target_vertices_sem, reduction="none")
    return dist.mean()


def fit_mano(
    rot,
    trans,
    shape,
    pose,
    mano_layer,
    epochs,
    optim_on,
    target_vertices,
    criteria_loss,
    sem_idx,
    lr,
    logger=None,
):
    logger = print if logger is None else logger.info

    sealed_vertices_sem_idx = np.load(
        "./external/v2a_code/body_models/sealed_vertices_sem_idx.npy",
        allow_pickle=True
    )
    params = []
    if "rot" in optim_on:
        params.append(rot)
    if "trans" in optim_on:
        params.append(trans)
    if "shape" in optim_on:
        params.append(shape)
    if "pose" in optim_on:
        params.append(pose)

    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, verbose=False)
    tol_lr = 1e-5
    # pbar = tqdm(range(epochs))
    for i in range(epochs):
        out = mano_layer(global_orient=rot, hand_pose=pose, betas=shape, transl=trans)
        hand_verts = out.vertices
        faces = mano_layer.faces.astype(np.int64)

        dist = criteria_loss(
            hand_verts.repeat(target_vertices.shape[0], 1, 1), target_vertices
        )
        vloss = dist.mean()
        eloss = edge_loss(faces, hand_verts, target_vertices)
        sloss = sem_loss(hand_verts, target_vertices, sealed_vertices_sem_idx, sem_idx)

        total_loss = vloss + eloss + 0.1 * sloss
        # total_loss = vloss + eloss
        # pbar.set_description("Fitting on %s, Loss: %.2e" % (optim_on, total_loss.data))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr < tol_lr:
            break
    # logger("Loss: %.2e" % (total_loss.data))

    # seal hand_verts
    centers = hand_verts[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
    hand_verts = torch.cat((hand_verts, centers), dim=1)
    
    return (rot, trans, shape, pose, hand_verts, float(total_loss))


def optimize_mano_shape(
    pred_vertices, mano_layers, optim_specs, iteration, init_params=None
):
    batch_size = 1  # 512  # 32
    ncomps = 45
    epoch_coarse = optim_specs["epoch_coarse"]
    epoch_fine = optim_specs["epoch_fine"]
    is_right = optim_specs["is_right"]
    save_mesh = optim_specs["save_mesh"]
    criterion = optim_specs["criterion"]
    torch.random.manual_seed(optim_specs["seed"])
    vis_dir = optim_specs["vis_dir"]
    sem_idx = optim_specs["sem_idx"]
    hand_type = "right" if is_right else "left"
    mano_layer = mano_layers[hand_type]

    target_vertices = pred_vertices.detach()
    target_vertex = target_vertices[0]
    faces = mano_layer.faces
    
    seal_faces = np.array(SEAL_FACES_R)
    if not is_right:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal

    
    if save_mesh:
        save_trimesh(
            target_vertex.cpu().detach().numpy(),
            faces,
            vis_dir + f"{iteration:04}_" + hand_type + "_target.ply",
        )
    faces = np.concatenate((faces, seal_faces))
    device = "cuda"
    mano_layer = mano_layer.to(device)

    # Model para initialization:
    shape = torch.rand(batch_size, 10).to(device)
    rot = torch.rand(batch_size, 3).to(device)
    pose = torch.zeros(batch_size, ncomps).to(device)
    trans = torch.rand(batch_size, 3).to(device)

    if init_params is not None:
        shape = init_params["betas"].view(1, 10).detach()
        rot = init_params["global_orient"].view(1, 3).detach()
        pose = init_params["hand_pose"].view(1, -1).detach()
        trans = init_params["transl"].view(1, 3).detach()

    shape.requires_grad_()
    rot.requires_grad_()
    pose.requires_grad_()
    trans.requires_grad_()

    start_vertices = mano_layer(
        global_orient=rot, hand_pose=pose, betas=shape, transl=trans
    ).vertices
    
    """
    if save_mesh:
        save_trimesh(
            start_vertices.detach().cpu().numpy()[0],
            faces,
            vis_dir + hand_type + "_hand_start.ply",
        )
    """

    # Optimize for global orientation
    rot, trans, shape, pose, hand_verts, _ = fit_mano(
        rot,
        trans,
        shape,
        pose,
        mano_layer,
        epoch_coarse,
        ["rot", "trans"],
        target_vertices,
        criterion,
        [],
        1e-1,
    )

    if save_mesh:
        save_trimesh(
            hand_verts.detach().cpu().numpy()[0],
            faces,
            vis_dir + f"{iteration:04}_" + hand_type + "_coarse.ply",
        )

    # Local optimization
    rot, trans, shape, pose, hand_verts, _ = fit_mano(
        rot,
        trans,
        shape,
        pose,
        mano_layer,
        epoch_fine,
        ["rot", "trans", "pose", "shape"],
        target_vertices,
        criterion,
        sem_idx,
        1e-2,
    )

    if save_mesh:
        save_trimesh(
            hand_verts.detach().cpu().numpy()[0],
            faces,
            vis_dir + f"{iteration:04}_" + hand_type + "_fine.ply",
        )
    return {"global_orient": rot, "hand_pose": pose, "betas": shape, "transl": trans}
