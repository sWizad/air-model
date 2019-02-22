import tensorflow as tf


def spatial_transformer_network(input_fmap, theta, out_dims=None,sigma2 = 0.1, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localization_net: takes the original image as input and outputs
      the parameters of the affine transformation that should be applied
      to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
      and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, H, W, C).
    - theta: affine transform tensor of shape (B, 6). Permits cropping,
      translation and isotropic scaling. Initialize to identity matrix.
      It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]

    # reshape theta to (B, 2, 3)
    theta = tf.reshape(theta, [B, 2, 3])
    #theta = tf.Print(theta, [theta[0,0,:],theta[0,1,:]])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    #inter_fmap = filterbank(input_fmap,sigma2)
    #out_fmap = bilinear_sampler(inter_fmap, x_s, y_s)

    #out_fmap, sum = filterbank2(input_fmap, x_s, y_s,sigma2)

    if out_dims:
        if out_H*out_W < input_fmap.shape[1]*input_fmap.shape[2]: #Big to small
            out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
        else: #small to big
            #inter_fmap = filterbank(input_fmap)
            #out_fmap = bilinear_sampler(inter_fmap, x_s, y_s)
            out_fmap = filterbank(input_fmap, x_s, y_s,sigma2) #bug rotation
    else:
        out_fmap = filterbank2(input_fmap, x_s, y_s,sigma2)
    # out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    #sampling_grid = tf.Print(sampling_grid, [tf.shape(sampling_grid)])
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))
    #x = tf.Print(x, [x[:,4,:]-x[:,5,:]])                    # Printing

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def filterbank(img, x,y, sigma2 = 0.5):
    """
    based on Alex Graves
    the rotation is bug
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    eps = 1e-8
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    H = img.shape[1]
    W = img.shape[2]
    num_batch = tf.shape(img)[0]
    color_ch = img.shape[3]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    A = tf.shape(x)[1]
    B = tf.shape(x)[2]

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))
    mu_x = tf.reshape(x[:,0:1,:],[-1,B,1])
    mu_y = y[:,:,0:1]


    a = tf.reshape(tf.cast(tf.range(H), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(W), tf.float32), [1, 1, -1])

    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2))
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)

    Fxt=tf.transpose(Fx,perm=[0,2,1])

    if color_ch == 3:
        glimpse0=tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,0],Fxt)),3)
        glimpse1=tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,1],Fxt)),3)
        glimpse2=tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,2],Fxt)),3)
        out = tf.concat([glimpse0,glimpse1,glimpse2],axis=3)
    else:
        out =tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,0],Fxt)),3)


    return out

def filterbank0(img, sigma2 = 0.1):
    """
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    eps = 1e-8
    #H = tf.shape(img)[1]
    #W = tf.shape(img)[2]
    H = img.shape[1]
    W = img.shape[2]
    num_batch = tf.shape(img)[0]
    color_ch = img.shape[3]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    grid_x = tf.reshape(tf.cast(tf.range(W), tf.float32), [1, -1])
    grid_y = tf.reshape(tf.cast(tf.range(H), tf.float32), [1, -1])
    mu_x = tf.reshape(grid_x, [-1, W, 1])
    mu_y = tf.reshape(grid_y, [-1, H, 1])
    mu_x = tf.tile(mu_x, tf.stack([num_batch, 1, 1]))
    mu_y = tf.tile(mu_y, tf.stack([num_batch, 1, 1]))

    a = tf.reshape(tf.cast(tf.range(H), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(W), tf.float32), [1, 1, -1])

    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2))
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)

    Fxt=tf.transpose(Fx,perm=[0,2,1])

    if color_ch == 3:
        glimpse0=tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,0],Fxt)),3)
        glimpse1=tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,1],Fxt)),3)
        glimpse2=tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,2],Fxt)),3)
        out = tf.concat([glimpse0,glimpse1,glimpse2],axis=3)
    else:
        out =tf.expand_dims(tf.matmul(Fy,tf.matmul(img[:,:,:,0],Fxt)),3)


    return out

def filterbank2(img, x, y, sigma2 = 0.1):
    """
    4x4 Gaussian interpolation
    Input--
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns--
    - out: interpolated images according to grids. Same size as grid.
    """
    eps = 1e-8
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 0, 'int32')
    max_x = tf.cast(W - 0, 'int32')
    zero = tf.zeros([], dtype='int32') -1

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    mu_x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    mu_y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(mu_x), 'int32') - 1
    y0 = tf.cast(tf.floor(mu_y), 'int32') - 1

    out = 0.0
    sum = 0.0
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    for i1 in range(4):
        for i2 in range(4):
            xx = tf.clip_by_value(x0+i1, zero, max_x)
            yy = tf.clip_by_value(y0+i2, zero, max_y)
            Ia = get_pixel_value(img, xx, yy)
            x1 = tf.cast(x0+i1, 'float32')
            y1 = tf.cast(y0+i2, 'float32')
            wa = tf.exp(-(tf.square(x1-mu_x)+tf.square(y1-mu_y))/(2*sigma2))
            wa = tf.expand_dims(wa, axis=3)
            out = out + wa*Ia
            sum = sum + wa
    out = out/tf.maximum(sum,eps)

    return out
