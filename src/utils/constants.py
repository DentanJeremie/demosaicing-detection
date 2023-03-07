from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
)

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