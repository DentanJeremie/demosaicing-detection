from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
)
FORGERY_WINDOWS_SIZE = 200
FORGERY_DETECTION_WINDOWS_SIZE = 100
JPEG_COMPRESSION_FACTORS = [None, 95, 90]
DEMOSAICING_ALGOS = {
    'bilinear': demosaicing_CFA_Bayer_bilinear,
    'malvar': demosaicing_CFA_Bayer_Malvar2004,
    'menon': demosaicing_CFA_Bayer_Menon2007,
}
PATERNS = [
    'RGGB',
    'BGGR',
    'GRBG',
    'GBRG',
]
DIAGONALS = {
    0:['RGGB','BGGR'],
    1:['GRBG','GBRG'],
}
ALGO_TO_INDEX = {
    algo:index
    for index, algo in enumerate(DEMOSAICING_ALGOS)
}
PATTERN_TO_INDEX = {
    pattern:index
    for index, pattern in enumerate(PATERNS)
}
PATTERN_TO_DIAG_INDEX = {
    pattern:diag_index
    for diag_index, (_, pattern_list) in enumerate(DIAGONALS.items())
    for pattern in pattern_list
}
ALGO_PATTERN_CONFIG = [
    (algo, pattern)
    for algo in DEMOSAICING_ALGOS
    for pattern in PATERNS
]