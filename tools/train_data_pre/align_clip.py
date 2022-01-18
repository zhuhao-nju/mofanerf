import json
import numpy as np
import os
import trimesh


def deleteFace(mesh, dis):
    meshDis = np.sqrt(np.sum(mesh.vertices ** 2, 1))
    vert_mask = (meshDis < dis)

    face_mask = vert_mask[mesh.faces].all(axis=1)
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def load_ori_mesh(fn):
    return trimesh.load(fn, resolver=None, split_object=False, group_material=False, skip_materials=False,
                        maintain_order=False, process=False)


# convert the old version index to the release version index
relList = np.loadtxt("./predef/order_new_old.txt")
relList = relList[np.argsort(relList[:, 1])]

order_dict = {}
for new, old in relList:
    order_dict[int(old)] = int(new)

# load maxDistance file, calculated by facescape bilinaer model
Max_distance_dict = np.load("./predef/maxDistance.npy")

expressionName = ["neutral", "smile", "mouth_stretch", "anger", "jaw_left", "jaw_right", "jaw_forward", "mouth_left",
                  "mouth_right", "dimpler",
                  "chin_raiser", "lip_puckerer", "lip_funneler", "sadness", "lip_roll", "grin", "cheek_blowing",
                  "eye_closed", "brow_raiser", "brow_lower"]

# read Rt scale
with open("predef/Rt_scale_dict.json", 'r') as f:
    Rt_scale_dict = json.load(f)

lackIndex = []
for id_idx in range(1, 2):
    for exp_idx in range(1, 21):

        mview_mesh = []
        dataPath = f'../data/models_raw/{id_idx}/{exp_idx}_{expressionName[exp_idx - 1]}/{exp_idx}_{expressionName[exp_idx - 1]}.obj'
        if os.path.exists(dataPath):
            print("loading {}".format(dataPath))
            try:
                mview_mesh = load_ori_mesh(dataPath)
                pass
            except Exception:
                print("Load error")
                lackIndex.append(id_idx)
                continue
                pass
        else:
            lackIndex.append(id_idx)
            continue

        id_idx_new = order_dict[id_idx]

        try:
            scale = Rt_scale_dict['%d' % id_idx_new]['%d' % exp_idx][0]
            pass
        except Exception:
            lackIndex.append(id_idx)
            print("Rt error")
            continue
            pass

        Rt = np.array(Rt_scale_dict['%d' % id_idx_new]['%d' % exp_idx][1])

        # align multi-view model to TU model
        mview_mesh.vertices *= scale
        mview_mesh.vertices = np.tensordot(Rt[:3, :3], mview_mesh.vertices.T, 1).T + Rt[:3, 3]

        # clip the model
        maxDis = Max_distance_dict[id_idx_new]
        mview_mesh = deleteFace(mview_mesh, maxDis)
        mview_mesh.visual.material.name = f'{exp_idx}_{expressionName[exp_idx - 1]}'

        # save clipped model
        os.makedirs(f"../data/models_out/{id_idx}", exist_ok=True)
        mview_mesh.export("../data/models_out/{}/{}_{}.obj".format(id_idx, exp_idx, expressionName[exp_idx - 1]))
        print("aligned mview_model saved to ../data/models_out/{}/{}_{}.obj".format(id_idx, exp_idx,
                                                                                    expressionName[exp_idx - 1]))
