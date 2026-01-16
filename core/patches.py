def extract_patches(img, corner_size=220, edge_thickness=60, grid=3):
    H, W = img.shape[:2]

    cs = min(corner_size, H//3, W//3)
    et = min(edge_thickness, H//10, W//10)

    corners = {
        "TL": img[:cs, :cs],
        "TR": img[:cs, W-cs:W],
        "BR": img[H-cs:H, W-cs:W],
        "BL": img[H-cs:H, :cs],
    }

    edges = {
        "top": img[:et, cs:W-cs],
        "bottom": img[H-et:H, cs:W-cs],
        "left": img[cs:H-cs, :et],
        "right": img[cs:H-cs, W-et:W],
    }

    surface = img[et:H-et, et:W-et]
    grid_patches = []

    h, w = surface.shape[:2]
    for r in range(grid):
        for c in range(grid):
            grid_patches.append(
                (r, c, surface[
                    int(r*h/grid):int((r+1)*h/grid),
                    int(c*w/grid):int((c+1)*w/grid)
                ])
            )

    return {
        "corners": corners,
        "edges": edges,
        "surface_grid": grid_patches
    }
