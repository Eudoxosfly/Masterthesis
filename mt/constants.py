import matplotlib

# This file contains all constants used in the project

# All specimen for analysis in general
ANALYSIS_SPECIMEN = ["BA38",
                     "BA37",
                     "BA36",
                     "AD77",
                     "AD76",
                     "AD68",
                     "AD67",
                     "AD66",
                     "AD64",
                     "AD63",
                     "AD52",
                     "AD51",
                     "AD50",
                     "AD48",
                     "AD45",
                     "AD43",
                     "AD42",
                     "AD40",
                     "AD38",
                     "AD36",
                     "AD33",
                     "AD32",
                     "AD29",
                     "AD28",
                     "AD21",
                     "AD20",
                     "AD19",
                     "AD18",
                     "AD17",
                     "AD16",
                     "AD15",
                     "AD14",
                     "AD12",
                     "AC15",
                     "AC11",
                     "AC14",
                     "BA07",
                     "BA05"]

# All specimen for high quality analysis
ANALYSIS_SPECIMEN2 = ["AD12",
                      "AD14",
                      "AD16",
                      "AD18",
                      "AD19",
                      "AD20",
                      "AD21",
                      "AD28",
                      "AD29",
                      "AD32",
                      "AD36",
                      "AD38",
                      "AD40",
                      "AD42",
                      "AD43",
                      "AD45",
                      "AD63",
                      "AD64",
                      "AD66",
                      "AD67", ]

# Custom color map for the segmentation of the bonding region
CMAP_MASK = matplotlib.colors.ListedColormap(((1, 1, 0.6),
                                              (0.2, 1, 1),
                                              (0.627, 0.627, 0.627)
                                              ))

# Scale factors for the 2D and 3D to calculate correct voxel sizes for areas and volumes from binned scans
SCALE_FACTOR_2D = 2 ** 2
SCALE_FACTOR_3D = 2 ** 3
