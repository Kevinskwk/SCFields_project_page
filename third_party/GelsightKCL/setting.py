def init():
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_, x0_l, y0_l, dx_l, dy_l, x0_r, y0_r, dx_r, dy_r
    RESCALE = 1
    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker (in original size)
    dx_, dy_: the horizontal and vertical interval between adjacent markers (in original size)
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """
    N_ = 8
    M_ = 6
    fps_ = 20

    x0_ = 55 / RESCALE
    y0_ = 70 / RESCALE
    dx_ = 90 / RESCALE
    dy_ = 90 / RESCALE

    x0_l = 25 / RESCALE
    y0_l = 82 / RESCALE
    dx_l = 72 / RESCALE
    dy_l = 72 / RESCALE

    x0_r = 60 / RESCALE
    y0_r = 70 / RESCALE
    dx_r = 72 / RESCALE
    dy_r = 72 / RESCALE
