from core.alignment import align_card_robust
from core.centering import compute_centering_final, centering_sensor_mode
from core.patches import extract_patches
from core.surface import predict_surface_patch, surface_score_from_patch_preds
from core.corners import corner_score_from_patches
from core.edges import edge_score_from_patches

def final_grade(c, k, e, s):
    g = min(c, k, e, s)
    if c <= 2 and min(k,e,s) >= 5:
        g = max(g, 3)
    return int(max(1, min(10, g)))

def grade_card(image_path):
    aligned, _, _ = align_card_robust(image_path)
    if aligned is None:
        return {"status": "alignment_failed"}

    (ratios,_),_ = compute_centering_final(aligned)
    centering = centering_sensor_mode(
        ratios["LR_ratio"], ratios["TB_ratio"]
    ) if ratios else 3

    patches = extract_patches(aligned)

    surface = surface_score_from_patch_preds([
        {"cls": predict_surface_patch(p)[0],
         "conf": predict_surface_patch(p)[1]}
        for _,_,p in patches["surface_grid"]
    ])

    corners = corner_score_from_patches(patches["corners"])
    edges = edge_score_from_patches(patches["edges"])

    return {
        "status":"ok",
        "final_grade": final_grade(centering, corners, edges, surface),
        "centering": centering,
        "corners": corners,
        "edges": edges,
        "surface": surface
    }
